#posturální stabilita


xytext = (0, 5 if value >= 0 else -5), textcoords = "offset points",
ha = "center", va = "bottom" if value >= 0 else "top", fontsize = 10)
fig.suptitle(POSTURE + ": rozdíl SD COP\nPojmenování barvy minus čtení slov", fontsize=14)
fig.savefig(output_dir / "rozdily_sd.png", dpi=220)
plt.close(fig)


def run_analysis(data_dir: Path = BASE_DIR / "Kistler", output_dir: Path = BASE_DIR / "vysledky",
                 make_plots: bool = True) -> Dict[str, Any]:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if not data_dir.is_dir():
        raise ValueError("Vstupní složka neexistuje: {}".format(data_dir))
    paths = sorted(data_dir.glob("*.txt"))
    infos = [parse_trial_name(path) for path in paths]
    validate_complete_trials(infos)
    # Všechna data nejprve ověříme a spočítáme. Při chybě nic neexportujeme.
    trial_rows, quality_rows, processed = [], [], []
    for path, info in zip(paths, infos):
        record = read_bioware(path)
        duration_s = len(record.time_s) / record.fs_hz
        if not np.isclose(duration_s, EXPECTED_DURATION_S, rtol=0, atol=1e-6):
            raise ValueError("{}: očekávám celý 30s pokus, záznam má {} s (N/fs).".format(path.name, duration_s))
        plate = select_loaded_plate(record)
        filtered_m = filter_cop(plate.cop_m, record.fs_hz)
        row = {"proband": info.participant, "podminka": info.condition, "opakovani": info.repetition,
               "soubor": path.name, "plosina": plate.index, "fs_hz": record.fs_hz,
               "pocet_vzorku": len(record.time_s), "delka_s": duration_s,
               "casovy_rozsah_s": float(record.time_s[-1] - record.time_s[0])}
        row.update(compute_sd_mm(filtered_m))
        trial_rows.append(row)
        for candidate in record.plates:
            fz = candidate.forces_n[:, 2]
            quality_rows.append({"soubor": path.name, "proband": info.participant,
                                 "podminka": info.condition, "opakovani": info.repetition,
                                 "plosina": candidate.index, "zarizeni": candidate.device,
                                 "vybrana": "ano" if candidate.index == plate.index else "ne",
                                 "fz_prumer_n": float(np.mean(fz)), "fz_median_n": float(np.median(fz)),
                                 "fz_min_n": float(np.min(fz)), "fz_max_n": float(np.max(fz)),
                                 "podil_vzorku_nad_prahem": float(np.mean(fz > LOAD_THRESHOLD_N))})
        processed.append((record, info, plate, filtered_m))
    means, differences = summarize_trials(trial_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pokusy.csv", trial_rows)
    write_csv(output_dir / "prumery.csv", means)
    write_csv(output_dir / "rozdily.csv", differences)
    write_csv(output_dir / "kontrola_dat.csv", quality_rows)
    write_xlsx(output_dir / "vysledky.xlsx", trial_rows, means, differences, quality_rows)
    settings = {
        "vytvoreno_utc": datetime.now(timezone.utc).isoformat(),
        "vstupni_slozka": str(data_dir.resolve()),
        "vstupni_soubory_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "pocet_probandu": len(PARTICIPANTS), "pocet_pokusu": len(paths), "podminky": list(CONDITIONS),
        "stoj": POSTURE, "instrukce_stoj": POSTURE_INSTRUCTIONS,
        "popis_podminek": CONDITION_LABELS,
        "osy": {"Ax": "ML", "Ay": "AP"},
        "jednotky": {"vstup_COP": "m", "vystup_SD_COP": "mm", "sily": "N"},
        "filtr": {"typ": "Butterworth, dolni propust", "rad_jednoho_pruchodu": FILTER_ORDER,
                  "mezni_frekvence_hz": CUTOFF_HZ,
                  "implementace": "scipy.signal.butter(output='sos') + sosfiltfilt(axis=0)",
                  "smer": "dopredne a zpetne", "padding": "vychozi scipy: padtype='odd', padlen=None",
                  "aplikovano_na": ["Ax", "Ay"], "Fz_filtrovano": False},
        "ddof_SD": SD_DDOF,
        "vyber_plosiny": {"pravidlo": "prave jedna plosina s medianem Fz > prah; vsechny jeji vzorky Fz > prah",
                          "prah_n": LOAD_THRESHOLD_N, "puvod_prahu": "prakticka kontrola kvality, neni ze zadani",
                          "contact_period_z_hlavicky_pouzit": False},
        "casove_okno": {"cela_delka_s": EXPECTED_DURATION_S, "definice_delky": "N / fs; casovy rozsah je (N-1) / fs",
                        "orez": False, "detrending": False, "odstranovani_vzorku": False},
        "agregace": "aritmeticky prumer dvou SD samostatnych opakovani; casove rady se nespojuji",
        "rozdil": "barevne minus cernobile; kladny = vetsi SD v podmince barevne",
        "porovnani_uloh": "pojmenovani barvy textu oproti cteni slov; obe podminky obsahuji kognitivni ulohu",
        "verze": {"python": platform.python_version(),
                  **{name: version(name) for name in ("numpy", "scipy", "matplotlib", "openpyxl")}},
        "grafy_vytvoreny": make_plots,
    }
    with (output_dir / "nastaveni.json").open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if make_plots:
        plots_dir = output_dir / "grafy"
        plots_dir.mkdir(exist_ok=True)
        for record, info, plate, filtered_m in processed:
            plot_trial(record, info, plate, filtered_m, plots_dir / (record.path.stem + ".png"))
        plot_summaries(means, differences, plots_dir)
    return {"trial_rows": trial_rows, "mean_rows": means, "difference_rows": differences,
            "quality_rows": quality_rows, "output_dir": output_dir}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "Kistler", help="Složka se 16 TXT exporty BioWare")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "vysledky", help="Výstupní složka")
    parser.add_argument("--no-plots", action="store_true", help="Uložit pouze tabulky a nastavení")
    args = parser.parse_args(argv)
    try:
        result = run_analysis(args.data_dir, args.output_dir, make_plots=not args.no_plots)
    except (OSError, ValueError) as exc:
        print("CHYBA: {}".format(exc), file=sys.stderr)
        print("Analýza nebyla dokončena. Případné starší výstupy nejsou výsledkem tohoto běhu.", file=sys.stderr)
        return 1
    print("Hotovo: 16 pokusů, 8 průměrů a 4 párové rozdíly.")
    print("Výstupy: {}".format(result["output_dir"].resolve()))
    print("Excel: vysledky.xlsx; grafy: podsložka grafy.")
    print("Rozdíl = pojmenování barvy minus čtení slov; hodnoty jsou v mm.")
    for row in result["difference_rows"]:
        print("Účastník {}: ML {:+.4f} mm, AP {:+.4f} mm".format(
            row["proband"], row["rozdil_ml_mm"], row["rozdil_ap_mm"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

