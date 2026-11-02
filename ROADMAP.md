# AstroProject — Master-Roadmap (Deep Sub-Sigma Discovery & Stellarium Control)

## Meilenstein 9: UI/UX 1cm Top-Statusbar & Live-Zeitreihen-Simulator (`gui/app.py`)
- [x] **1cm Fixed Top-Statusbar:** Einzeilige, flache Statusleiste am oberen Bildschirmrand (CPU %, RAM %, GPU-Name, VRAM GB, Device) ohne verschenkten Platz über dem Haupttitel.
- [x] **Synthetischer Low-SNR Zeitreihen-Generator:** Button in Tab 2 zum Erzeugen einer 5-Frame-FITS-Serie mit statischen Feldsternen und einem im Einzelbild fast unsichtbaren Asteroiden (`SNR ~ 1.0 - 1.5` im Einzelbild) inkl. Vorher/Nachher-Plot.

## Meilenstein 10: KBMOD Psi/Phi Sub-Sigma GPU-Engine & 3-Kanal RealBogusCNN (`velocity_search.py`, `classifier.py`)
- [ ] **Pre-Shift PSF Matched-Filtering (Psi/Phi-Methode):** Einmalige Faltung der Einzelbilder mit der PSF auf der GPU *vor* der Geschwindigkeitsschleife zur Maximierung des SNR schwacher Punktquellen.
- [ ] **Trimmed-Likelihood & 3D GPU Peak-Finder:** Unterdrückung von Einzelbild-Ausreißern (Cosmic Rays) durch Abschneiden des höchsten Einzel-Frames entlang der Zeitachse (`torch.topk`) und 3D-Non-Maximum-Suppression (`F.max_pool3d`) über `[v, Y, X]` direkt im VRAM. Erlaubt Detektionen bis `2.2 sigma` ohne CPU-Flaschenhals.
- [ ] **3-Kanal-CNN (`[Mean, Median, Variance]`) & Real-Background Implanting:** Erweiterung des `RealBogusCNN` auf 3 Kanäle sowie automatischer Trainings-Generator (`calibrate_on_field`), der echte Feldsterne als empirische PSF klont und als schwache Fake-Asteroiden gegen echte Hintergrund-Artefakte trainiert (`models/real_bogus_3ch.pt`).

## Meilenstein 11: Single-Pass WCS-Registrierung & Multi-HDU FITS (`wcs_utils.py`, `fits_loader.py`)
- [x] **Kombinierte WCS- & Kinematik-Transformation:** Verschmelzung von Teleskop-Drift-Korrektur (`origin=0`) und `(vx, vy)`-Shift in einem einzigen flusserhaltenden `F.grid_sample(..., align_corners=False)` Aufruf.
- [x] **Robuster Multi-HDU FITS-Loader:** Unterstützung für komprimierte Archiv-FITS (`.fits.fz`) und mehrdimensionale HDUs.

## Meilenstein 12: Bahnbestimmung, Known-Object Cross-Match & MPC-Export (`orbit.py`, `catalog_match.py`)
- [ ] **Kepler-Bahnschätzung (IOD):** Berechnung von `(RA, Dec)` zu `t_start, t_mid, t_end` sowie Schätzung der heliozentrischen Bahnelemente (`a, e, i, Omega, omega, M`) via Väisälä/Gauß-Methode.
- [ ] **Known-Object Cross-Match (SkyBoT / MPC):** Abgleich gefundener Koordinaten und Geschwindigkeiten gegen bekannte Ephemeriden, um echte Neuentdeckungen (`UNKNOWN CANDIDATE`) sofort hervorzuheben.
- [ ] **Multi-Stamp Blink-Viewer & MPC-Export:** Visueller Vergleich (`Mean`, `Median`, `Even`, `Odd`, Einzel-Frames) in der GUI sowie Export im offiziellen 80-Spalten-MPC-/ADES-Format.

## Meilenstein 13: Autonomes Nachbarfeld-Tracking & Stellarium-Bridge (`tracker.py`, `stellarium.py`)
- [ ] **Bildrand-Extrapolation & Auto-Fetch:** Vorhersage von Austrittszeitpunkt `t_exit` und Zielkoordinaten `(RA_next, Dec_next)` am Bildrand mit automatischem Nachladen angrenzender FITS-Cutouts.
- [ ] **Stellarium RemoteControl Integration:** Direkte Anbindung an `http://localhost:8090/api/` zum automatischen Setzen der Beobachtungszeit und Schwenken der Stellarium-Kamera auf gefundene Kandidaten sowie Export berechneter Orbits für Stellarium.
