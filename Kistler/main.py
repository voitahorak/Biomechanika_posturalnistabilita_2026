#postrurální stabilita v tandemovém stoji


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  #grafy se ulozi do png obrazku
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


#Nastavení a výběr složky
fvz = 200  #vzorkovací frekvence
prah_sily = 50  #naše kontrolní hranice zatížené plošiny
podminky = {"cernobile": "Čtení slov", "barevne": "Pojmenování barvy"}
metriky = ["SD ML [mm]", "SD AP [mm]"]
filtr = butter(4, 10, btype="lowpass", fs=fvz, output="sos")

#složku Kistler najdeme automaticky, takže nepotřebujeme výběrové okno
adresar_skriptu = Path(__file__).resolve().parent
if adresar_skriptu.name.lower() == "kistler":
    slozka = adresar_skriptu  #main.py už je přímo ve složce Kistler
else:
    slozka = adresar_skriptu / "Kistler"  #jinak hledáme Kistler vedle main.py
if not slozka.is_dir():
    raise FileNotFoundError(f"Nenašel jsem složku Kistler: {slozka}")

#výsledky uložíme vedle složky Kistler
vystup = slozka.parent / "vysledky_stabilita"
grafy = vystup / "grafy"
vysledky = []
signaly = []
print(f"Načítám data ze složky: {slozka}", flush=True)


#Výpočet pro všechny osoby
for ucastnik in range(1, 5):
    for podminka, popis in podminky.items():
        for pokus in (1, 2):
            pripona = "" if pokus == 1 else "_2"
            soubor = slozka / f"Proband_{ucastnik}_{podminka}{pripona}.txt"
            if not soubor.is_file():
                raise FileNotFoundError(f"Chybí pokus: {soubor}")

            ###
            hlavicka = soubor.read_text(encoding="utf-8-sig").splitlines()[:19]
            kanaly = ["Fx", "Fy", "Fz", "Ax", "Ay"] * 2
            jednotky = ["N", "N", "N", "m", "m"] * 2
            if (len(hlavicka) < 19 or hlavicka[17].split("\t")[1:] != kanaly
                    or hlavicka[18].split("\t")[1:] != jednotky):
                raise ValueError(f"{soubor.name}: jiná hlavička nebo jednotky exportu.")
            data = pd.read_csv(soubor, sep="\t", skiprows=19, header=None)
            if data.shape != (6000, 11) or not np.isfinite(data.to_numpy()).all():
                raise ValueError(f"{soubor.name}: očekávám 6000 řádků a 11 číselných sloupců.")
            cas = data.iloc[:, 0].to_numpy()
            if not np.allclose(np.diff(cas), 1 / fvz, rtol=0, atol=1.1e-6):
                raise ValueError(f"{soubor.name}: časová osa neodpovídá 200 Hz.")

            #Fz je ve sloupcích 3 a 8
            fz = data.iloc[:, [3, 8]].to_numpy()
            zatizene = np.flatnonzero(np.median(fz, axis=0) > prah_sily)
            if len(zatizene) != 1:
                raise ValueError(f"{soubor.name}: musí být zatížená právě jedna plošina.")
            plosina = int(zatizene[0])  #0->prvni plosina, 1->druha plosina
            if np.min(fz[:, plosina]) <= prah_sily:
                raise ValueError(f"{soubor.name}: zatížení během pokusu kleslo pod kontrolní práh.")

            #Ax = ML, Ay = AP. Na první plošině sloupce 4/5, na druhé 9/10.
            cop = data.iloc[:, [4 + 5 * plosina, 5 + 5 * plosina]].to_numpy() * 1000
            cop_filtr = sosfiltfilt(filtr, cop, axis=0)  #obousměrná filtrace, osy zvlášť
            sd_ml, sd_ap = np.std(cop_filtr, axis=0, ddof=1)  #výběrová SD v mm
            vysledky.append({
                "Účastník": ucastnik, "Podmínka": popis, "Pokus": pokus,
                "SD ML [mm]": sd_ml, "SD AP [mm]": sd_ap,
                "Plošina": plosina + 1, "Soubor": soubor.name,
            })
            signaly.append((soubor.stem, ucastnik, popis, pokus, cas, cop, cop_filtr, fz))
            print(f"Zpracováno {len(vysledky)}/16: {soubor.name}", flush=True)  #průběžná kontrola zpracování


#Tabulky SD pokusu
tabulka = pd.DataFrame(vysledky)
prumery = tabulka.groupby(["Účastník", "Podmínka"], as_index=False)[metriky].mean()
cteni = prumery[prumery["Podmínka"] == podminky["cernobile"]].set_index("Účastník")
barvy = prumery[prumery["Podmínka"] == podminky["barevne"]].set_index("Účastník")
rozdily = (barvy[metriky] - cteni[metriky]).rename(columns={
    "SD ML [mm]": "Rozdíl ML [mm]", "SD AP [mm]": "Rozdíl AP [mm]"
}).reset_index()  #znaménko ponecháváme: barvy minus čtení slov

#Výstupy
print("Ukládám Excel a grafy...", flush=True)
grafy.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(vystup / "vysledky.xlsx", engine="openpyxl") as excel:
    prumery.to_excel(excel, sheet_name="Prumery", index=False)
    rozdily.to_excel(excel, sheet_name="Rozdily", index=False)
    tabulka.to_excel(excel, sheet_name="Pokusy", index=False)
    for list_excel in excel.sheets.values():
        list_excel.freeze_panes = "A2"
        list_excel.auto_filter.ref = list_excel.dimensions
        for sloupec in list_excel.columns:
            sirka = max(len(str(bunka.value)) for bunka in sloupec) + 2
            list_excel.column_dimensions[sloupec[0].column_letter].width = sirka
            for bunka in sloupec[1:]:
                if isinstance(bunka.value, float):
                    bunka.number_format = "0.000"


#Průběhy před a po filtrací
for nazev, ucastnik, popis, pokus, cas, cop, cop_filtr, fz in signaly:
    fig, osy = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for smer, osa in enumerate(osy[0]):
        osa.plot(cas, cop[:, smer], color="0.7", label="Původní", linewidth=0.7)
        osa.plot(cas, cop_filtr[:, smer], label="Filtrovaný", linewidth=0.9)
        osa.set(xlabel="Čas [s]", ylabel=f"COP {('ML', 'AP')[smer]} [mm]")
    osy[1, 0].plot(cop_filtr[:, 0], cop_filtr[:, 1], linewidth=0.7, label="Filtrovaný COP")
    osy[1, 0].set(xlabel="COP ML [mm]", ylabel="COP AP [mm]", title="Stabilogram")
    osy[1, 0].set_aspect("equal", adjustable="box")
    osy[1, 1].plot(cas, fz[:, 0], label="Plošina 1", linewidth=0.7)
    osy[1, 1].plot(cas, fz[:, 1], label="Plošina 2", linewidth=0.7)
    osy[1, 1].axhline(prah_sily, color="gray", linestyle="--", label="Kontrolní práh 50 N")
    osy[1, 1].set(xlabel="Čas [s]", ylabel="Fz [N]", title="Kontrola zatížení")
    for osa in osy.flat:
        osa.legend(fontsize=8)
        osa.grid(alpha=0.2)
    fig.suptitle(f"Tandemový stoj: účastník {ucastnik}, pokus {pokus}\n{popis}")
    fig.savefig(grafy / f"{nazev}.png", dpi=200)
    plt.close(fig)


#Souhrnné grafy
fig, osy = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
for osa, metrika in zip(osy, metriky):
    prumery.pivot(index="Účastník", columns="Podmínka", values=metrika).plot.bar(ax=osa, rot=0)
    osa.set(ylabel=metrika, ylim=(0, prumery[metriky].max().max() * 1.15))
    osa.grid(axis="y", alpha=0.2)
fig.suptitle("Tandemový stoj: průměr SD dvou opakování")
fig.savefig(grafy / "prumerne_sd.png", dpi=200)
plt.close(fig)

osa = rozdily.set_index("Účastník").plot.bar(rot=0, figsize=(9, 5))
osa.axhline(0, color="black", linewidth=1)
osa.set(ylabel="Rozdíl SD COP [mm]", title="Tandemový stoj: pojmenování barvy minus čtení slov")
osa.grid(axis="y", alpha=0.2)
osa.figure.tight_layout()
osa.figure.savefig(grafy / "rozdily_sd.png", dpi=200)
plt.close(osa.figure)

print(rozdily.round(3).to_string(index=False))
print(f"Hotovo: 16 pokusů, Excel a 18 grafů. Výstupy: {vystup}")
