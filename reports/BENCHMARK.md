# Benchmark — does more data fix the overfitting?

**Lane WS-11 · wave 3.** Produced by `src/evaluate.py`; regenerate with the commands under *Method*.

Wave 2 ended with one loud, unresolved claim in `STATUS.md`: *"a 135-name dataset cannot support a 2-layer, 256-wide LSTM. This is an argument for more data."* Wave 3's data lanes then took `data/` to **30 datasets / 27,226 names**, growing four thin files in place. This report tests the claim.

Coverage: **30 of the 30 datasets** in `data/`, i.e. all of them. The data lanes were still committing while these runs were in flight, so treat the dataset count as a snapshot.

**Answer: yes to "more data makes a better model" — held-out loss improved monotonically with dataset size in every domain tested. No to "more data fixes the overfitting" — the train/val gap at the best epoch did not shrink, and on the largest dataset it widened. What data does fix is the catastrophic 300-epoch collapse wave 2 documented, which shrank by 36%.**

---

## 1. The controlled measurement: a within-dataset size ladder

Comparing *different* datasets by their train/val gap cannot answer the question, because datasets differ in domain entropy as much as in size: `aircraft` reaches a val loss of 0.774 and `motorcycle_brands` reaches 2.731, and most of that spread is how predictable each naming style is, not how badly each overfits. So the primary experiment varies **only** size:

- one validation set per dataset, split off once (15%, seed 1337) and **held fixed across every arm** — every loss below is measured on the identical names;
- the remaining pool shuffled once, then trained on nested prefixes of it, so each smaller arm is a strict subset of every larger arm;
- vocabulary built from the full file at every arm, so model dimensions never move;
- identical config, identical seed, patience 20 everywhere.

| dataset | train names | best val loss | best epoch | train NLL | held-out NLL | gap | novelty | near-dup ≤1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `motorcycle_brands` | 54 | **3.163** | 18 | 2.543 | 3.149 | +0.607 | 100.0% | 1.7% |
| `motorcycle_brands` | 130 | **2.912** | 24 | 2.161 | 2.927 | +0.766 | 100.0% | 0.0% |
| `motorcycle_brands` | 263 | **2.719** | 12 | 2.281 | 2.731 | +0.449 | 100.0% | 0.0% |
| `car_manufacturers` | 135 | **2.768** | 14 | 2.435 | 2.769 | +0.334 | 100.0% | 3.3% |
| `car_manufacturers` | 250 | **2.661** | 12 | 2.326 | 2.661 | +0.335 | 100.0% | 3.3% |
| `car_manufacturers` | 502 | **2.563** | 9 | 2.381 | 2.565 | +0.184 | 100.0% | 11.7% |
| `spacecraft` | 230 | **2.688** | 21 | 1.651 | 2.686 | +1.035 | 100.0% | 0.0% |
| `spacecraft` | 504 | **2.484** | 14 | 1.621 | 2.485 | +0.864 | 100.0% | 6.7% |
| `car_models` | 218 | **2.687** | 17 | 2.215 | 2.677 | +0.462 | 100.0% | 1.7% |
| `car_models` | 400 | **2.544** | 10 | 2.301 | 2.537 | +0.237 | 100.0% | 1.7% |
| `car_models` | 700 | **2.455** | 11 | 1.982 | 2.448 | +0.466 | 100.0% | 8.3% |
| `car_models` | 1035 | **2.406** | 9 | 2.005 | 2.405 | +0.399 | 100.0% | 5.0% |
| `english_words` | 500 | **2.400** | 7 | 2.107 | 2.383 | +0.276 | 98.3% | 11.7% |
| `english_words` | 1200 | **2.296** | 7 | 1.833 | 2.278 | +0.444 | 100.0% | 6.7% |
| `english_words` | 2500 | **2.182** | 7 | 1.667 | 2.166 | +0.500 | 93.3% | 36.7% |

### What the ladder shows

- **`motorcycle_brands`** — 54 → 263 training names (4.9×): best val loss 3.163 → 2.719 (**14% better**), gap +0.607 → +0.449, best epoch 18 → 12.
- **`car_manufacturers`** — 135 → 502 training names (3.7×): best val loss 2.768 → 2.563 (**7% better**), gap +0.334 → +0.184, best epoch 14 → 9.
- **`spacecraft`** — 230 → 504 training names (2.2×): best val loss 2.688 → 2.484 (**8% better**), gap +1.035 → +0.864, best epoch 21 → 14.
- **`car_models`** — 218 → 1035 training names (4.7×): best val loss 2.687 → 2.406 (**10% better**), gap +0.462 → +0.399, best epoch 17 → 9.
- **`english_words`** — 500 → 2500 training names (5.0×): best val loss 2.400 → 2.182 (**9% better**), gap +0.276 → +0.500, best epoch 7 → 7.

**Two different answers, and they disagree — which is the most useful thing in this report.**

1. **Held-out loss: more data wins, every time.** Best val loss fell monotonically with training-set size in **5 of 5** domains — every single arm, no exceptions, with the validation names held literally constant. If the question is "does more data make a better model", the answer is an unambiguous yes.
2. **Train/val gap at the best epoch: no, it does not shrink.** Only 1 of 5 domains showed a monotone decrease in the gap, and on `english_words` it moved the *wrong* way (+0.276 at 500 names → +0.500 at 2500). More data buys a better model, not a model that overfits less per epoch.

Those two facts are consistent: early stopping is already doing the work the gap would otherwise expose. When the run halts at epoch 7–24, it halts before the divergence, so the surviving gap mostly reflects how hard the domain is rather than how much data there was. Where data *does* visibly reduce overfitting is in the uncontrolled 300-epoch regime — section 2.

Also read the *shape*: the returns flatten. `car_models` gains 0.143 nats going 218 → 400 names and only 0.049 going 700 → 1035, and the best epoch drifts earlier as data grows rather than later. That is not the signature of a model that is merely data-starved. The honest reading is that wave 2's diagnosis was **half** the story: these datasets were too small, *and* a 2-layer 256-wide LSTM is too large for them. Data alone will not close the remaining distance — WS-9's smaller architectures are the other half of the experiment.

---

## 2. Head-to-head with wave 2, under wave 2's exact protocol

Wave 2's 6.15-nat gap was measured at **epoch 300 with no early stopping**. Every other number in this report comes from early-stopped runs with best-epoch weights restored, whose gaps are 0.07–0.79 largely *because* the run stops before the divergence. Putting those two numbers in one table would manufacture a dramatic improvement out of a methodology change. So this section re-runs wave 2's protocol verbatim — 300 epochs, 15% holdout, patience 0, stock config — on today's files.

| dataset | names | train NLL @300 | val NLL @300 | gap @300 | best val | best epoch | val degradation from best |
|---|---:|---:|---:|---:|---:|---:|---:|
| `aircraft` — **wave 2 (control)** | 435 | n/r | 1.02 | 0.64 | 0.76 | 24-26 | +35% |
| `aircraft` — wave 3 rerun, **file unchanged** | 435 | 0.38 | 1.08 | 0.71 | 0.809 | 21 | +34% |
| `car_manufacturers` — **wave 2** | 159 | 0.72 | 6.87 | **6.15** | 2.98 | 12-19 | +130% |
| `car_manufacturers` — **wave 3, grown file** | 590 | 0.84 | 4.79 | **3.95** | 2.582 | 10 | +86% |
| `spacecraft` — wave 3 | 593 | 0.67 | 4.01 | **3.34** | 2.481 | 15 | +62% |
| `motorcycle_brands` — wave 3 | 309 | 0.76 | 5.46 | **4.70** | 2.762 | 10 | +98% |

**`aircraft` is the protocol control.** The wave-3 data lanes did not touch that file, so its numbers should reproduce wave 2's — and they do: gap 0.64 → 0.71, degradation +35% → +34%, best epoch 24-26 → 21. That is what licenses reading the `car_manufacturers` row below as a data effect rather than a measurement effect.

**`car_manufacturers`, 159 → 590 names, same protocol:** the 300-epoch gap fell 6.15 → 3.95 nats/char (**36% smaller**), best val loss fell 2.98 → 2.582 (**13% better**), and the post-best collapse shrank from +130% to +86%.

That is a real improvement, and it is the result that justifies the wave-3 data effort. It is also **not a fix**. A 3.95-nat gap still means held-out characters are ~52× less likely under the model than training characters, and the best epoch moved *earlier* (12-19 → 10), not later — so ~97% of the stock 300-epoch budget is still actively harmful on this dataset.

One caveat this section cannot escape: wave 2 measured the *original* 159-name file and this measures the *grown* 590-name file. They share a name and a domain, not a composition. Section 1's ladder is the version of this comparison that controls for that, and it agrees.

---

## 3. The library: every dataset, one checkpoint each

All 30 datasets, trained with `--val-fraction 0.15` and patience 20, best-epoch weights restored, then evaluated identically (200 names, temperature 1.0, plain decoding, seed 0).

**Read this as a catalog, not as evidence about dataset size.** The gap column here is the gap *at the best epoch*, and across datasets it is dominated by domain entropy rather than by size — `aircraft` names share model-number structure and score 0.44/0.94, while `motorcycle_brands` are short unrelated words. Section 1 is where the size question is actually answered.

| dataset | names | train/val | train NLL | held-out NLL | gap | best val loss | best epoch | epochs run |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `motorcycle_brands` | 309 | 263/46 | 2.441 | 2.750 | +0.309 | 2.731 | 10 | 30/180 |
| `racehorses` | 355 | 302/53 | 1.662 | 2.316 | +0.654 | 2.298 | 20 | 40/180 |
| `motorcycles` | 359 | 305/54 | 0.566 | 1.357 | +0.791 | 1.344 | 20 | 40/180 |
| `paint_colors` | 391 | 332/59 | 1.017 | 1.593 | +0.576 | 1.577 | 13 | 33/180 |
| `locomotives` | 395 | 336/59 | 1.942 | 2.476 | +0.534 | 2.477 | 11 | 31/180 |
| `craft_beers` | 398 | 338/60 | 0.738 | 1.385 | +0.647 | 1.383 | 20 | 40/180 |
| `tech_startups` | 400 | 340/60 | 2.381 | 2.717 | +0.336 | 2.718 | 11 | 31/180 |
| `cheeses` | 421 | 358/63 | 2.042 | 2.451 | +0.409 | 2.450 | 12 | 32/180 |
| `aircraft` | 435 | 370/65 | 0.440 | 0.943 | +0.503 | 0.774 | 23 | 43/180 |
| `mushrooms` | 441 | 375/66 | 0.899 | 1.703 | +0.805 | 1.805 | 16 | 36/180 |
| `cocktails` | 459 | 390/69 | 1.933 | 2.440 | +0.507 | 2.464 | 11 | 31/180 |
| `dog_breeds` | 461 | 392/69 | 1.048 | 1.897 | +0.849 | 1.935 | 16 | 36/180 |
| `typefaces` | 463 | 394/69 | 1.710 | 2.373 | +0.663 | 2.372 | 14 | 34/180 |
| `sailing_ships` | 468 | 398/70 | 2.147 | 2.423 | +0.277 | 2.301 | 10 | 30/180 |
| `stars_constellations` | 503 | 428/75 | 1.720 | 2.180 | +0.459 | 2.198 | 12 | 32/180 |
| `mountains` | 523 | 445/78 | 1.861 | 2.170 | +0.309 | 2.187 | 10 | 30/180 |
| `board_games` | 585 | 497/88 | 1.828 | 2.485 | +0.658 | 2.494 | 13 | 33/180 |
| `car_manufacturers` | 590 | 502/88 | 2.282 | 2.563 | +0.280 | 2.566 | 11 | 31/180 |
| `spacecraft` | 593 | 504/89 | 1.748 | 2.485 | +0.737 | 2.484 | 12 | 32/180 |
| `minerals_gems` | 624 | 530/94 | 1.717 | 1.925 | +0.208 | 1.927 | 10 | 30/180 |
| `plants_flowers` | 634 | 539/95 | 1.840 | 2.250 | +0.410 | 2.248 | 10 | 30/180 |
| `perfumes` | 639 | 543/96 | 1.971 | 2.434 | +0.463 | 2.433 | 11 | 31/180 |
| `metal_bands` | 663 | 564/99 | 2.053 | 2.596 | +0.544 | 2.523 | 10 | 30/180 |
| `video_games` | 753 | 640/113 | 1.641 | 2.305 | +0.663 | 2.280 | 12 | 32/180 |
| `greek_myth` | 755 | 642/113 | 1.945 | 2.149 | +0.204 | 2.147 | 10 | 30/180 |
| `birds` | 863 | 734/129 | 0.755 | 1.645 | +0.890 | 1.431 | 15 | 35/120 |
| `car_models` | 1218 | 1035/183 | 1.919 | 2.397 | +0.478 | 2.401 | 10 | 30/120 |
| `world_cities` | 1691 | 1437/254 | 2.171 | 2.452 | +0.281 | 2.452 | 8 | 28/120 |
| `pharma_drugs` | 2223 | 1890/333 | 1.568 | 1.996 | +0.428 | 2.026 | 10 | 30/120 |
| `english_words` | 8631 | 7336/1295 | 1.606 | 1.988 | +0.383 | 2.006 | 6 | 26/60 |

### Cross-dataset correlations — weak, and confounded

| relationship | statistic | value |
|---|---|---|
| log10(size) vs. gap at best epoch | Pearson r | -0.218 |
| size vs. gap at best epoch | Spearman ρ | -0.222 |
| size vs. best val loss | Spearman ρ | +0.005 |
| size vs. novelty | Spearman ρ | +0.008 |
| size vs. near-duplicate ≤1 | Spearman ρ | +0.041 |

Mean gap under 500 training names: +0.546 (n=17). At or above 500: +0.459 (n=13). In the expected direction, and **not something to lean on**: n=30, one seed each, and size is confounded with domain regularity throughout. The ladder in section 1 is the same question asked properly.

---

## 4. Output quality

Novelty = distinct generated names absent from the training split. Near-duplicate ≤1 = within one edit of a training name — the memorization novelty alone misses. Plausibility ratio ≈ 1.00 means generated names are as bigram-typical as real ones.

| dataset | names | novelty | near-dup ≤1 | near-dup ≤2 | uniqueness | plaus. ratio | mean len |
|---|---:|---:|---:|---:|---:|---:|---:|
| `motorcycle_brands` | 309 | 100.0% | 2.5% | 23.5% | 100.0% | 1.09 | 6.4 |
| `racehorses` | 355 | 100.0% | 0.0% | 0.0% | 100.0% | 1.05 | 11.3 |
| `motorcycles` | 359 | 80.2% | 37.1% | 50.8% | 98.5% | 1.01 | 13.7 |
| `paint_colors` | 391 | 85.3% | 46.6% | 67.0% | 95.5% | 1.02 | 9.1 |
| `locomotives` | 395 | 100.0% | 0.5% | 7.0% | 100.0% | 1.03 | 10.4 |
| `craft_beers` | 398 | 96.0% | 6.1% | 10.6% | 99.0% | 1.01 | 18.3 |
| `tech_startups` | 400 | 100.0% | 1.5% | 15.0% | 100.0% | 1.06 | 7.0 |
| `cheeses` | 421 | 100.0% | 0.5% | 2.5% | 100.0% | 1.07 | 10.9 |
| `aircraft` | 435 | 63.1% | 65.8% | 77.0% | 93.5% | 1.01 | 15.9 |
| `mushrooms` | 441 | 99.0% | 3.0% | 7.6% | 99.0% | 1.01 | 14.0 |
| `cocktails` | 459 | 100.0% | 1.0% | 1.5% | 100.0% | 1.06 | 11.7 |
| `dog_breeds` | 461 | 99.5% | 1.0% | 4.0% | 100.0% | 0.99 | 15.6 |
| `typefaces` | 463 | 100.0% | 1.0% | 11.5% | 100.0% | 1.03 | 9.1 |
| `sailing_ships` | 468 | 100.0% | 0.5% | 6.0% | 100.0% | 1.05 | 10.5 |
| `stars_constellations` | 503 | 99.0% | 8.0% | 34.5% | 100.0% | 1.03 | 7.6 |
| `mountains` | 523 | 100.0% | 1.5% | 8.5% | 100.0% | 1.03 | 11.0 |
| `board_games` | 585 | 100.0% | 0.0% | 6.5% | 100.0% | 1.04 | 11.3 |
| `car_manufacturers` | 590 | 100.0% | 8.0% | 30.5% | 100.0% | 1.06 | 6.8 |
| `spacecraft` | 593 | 99.5% | 4.0% | 23.5% | 100.0% | 1.06 | 8.4 |
| `minerals_gems` | 624 | 99.5% | 2.5% | 28.0% | 100.0% | 1.05 | 9.3 |
| `plants_flowers` | 634 | 100.0% | 0.0% | 5.5% | 100.0% | 1.03 | 11.8 |
| `perfumes` | 639 | 100.0% | 2.5% | 13.0% | 100.0% | 1.02 | 9.2 |
| `metal_bands` | 663 | 100.0% | 0.0% | 5.0% | 100.0% | 1.05 | 10.0 |
| `video_games` | 753 | 100.0% | 1.0% | 4.0% | 100.0% | 1.04 | 11.6 |
| `greek_myth` | 755 | 100.0% | 9.0% | 45.5% | 100.0% | 1.05 | 6.8 |
| `birds` | 863 | 96.0% | 5.5% | 8.0% | 100.0% | 1.01 | 14.1 |
| `car_models` | 1218 | 97.0% | 20.0% | 51.0% | 100.0% | 1.02 | 6.5 |
| `world_cities` | 1691 | 99.5% | 3.5% | 19.5% | 100.0% | 1.01 | 8.8 |
| `pharma_drugs` | 2223 | 100.0% | 0.0% | 6.5% | 100.0% | 0.98 | 9.8 |
| `english_words` | 8631 | 77.4% | 55.3% | 82.4% | 99.5% | 0.99 | 6.4 |

## 5. Samples

Five per dataset, straight from the runs above. No metric replaces reading them.

| dataset | names | samples |
|---|---:|---|
| `motorcycle_brands` | 309 | Rict, Segter, Kasunetd, Nanokcho, Aiin |
| `racehorses` | 355 | Solicteol Sam, Skhim Tifron, Gane Cellia, Scarinine Ver, Hoot Oruple |
| `motorcycles` | 359 | Honda Trail, Harley-Davadsy, Ninja Mojo, MV Agusta Rivale, Ducati Diapilen |
| `paint_colors` | 391 | Bemenis, Babem Blukt, Bemidnight, erco Blqie, Bemidjuaze |
| `locomotives` | 395 | Slorn of Jalle, Bomdwinthlren Hanl, Ymy Hil, Bondenzander, Hlaldy |
| `craft_beers` | 398 | Hopworpd Brees, Wper Edofte, Ma2nal ano Choat, Mash 5isc Brewing Company Tanky Ond, Diamond Rittiy |
| `tech_startups` | 400 | Adusta, Yipp, Iyleroal, EntibB, OpphB Sceacho |
| `cheeses` | 421 | Rocte dole Giderin, Hofgres  do Vlentincave, Asgiatgirog, Dunnwerlo, Bayeuzla |
| `aircraft` | 435 | Bombardier Learjet 55, Piper eamhon, Boeing 787-9, Kecsna Citatiot CJ2, Karshing Piler ClRga Exper Pone |
| `mushrooms` | 441 | Snincerwood Ealthanshorn, Pabittert Saly Crust, Waines Afushroom, Beychake, Stripk Porach |
| `cocktails` | 459 | Blond aol, Piiderin Doar, Cho Doeslang, Fuch Jaskiangwa, Soothardila |
| `dog_breeds` | 461 | Baston Cranche Kerter, Janich Spiin, Blue de BrajFer, GerrFolan Windong, Gaker Spaniel |
| `typefaces` | 463 | Sabriar, Courkelad, Bankron, Pliplerter, Pollhitina |
| `sailing_ships` | 468 | HMS Aerwor, Skrloz, Bankar, Hyboule, Sevas el Con Wausber foles Pewte |
| `stars_constellations` | 503 | Talicar, Cakikala, Dycin Keltair, Alteran, Saraxis |
| `mountains` | 523 | Mount Kolu, Hiabain Paak, Han Lorba, Mount Weajugia, Mount Etgaru |
| `board_games` | 585 | Rixi, Sdice Induun, Bolama, Tchravein, Blice the Rinn  |
| `car_manufacturers` | 590 | Rictond, Wenanduvert, Kamanichi, Aiis, Binter |
| `spacecraft` | 593 | Surpead, Sorek, Seraviosat, Arnarer  Inelai, SXSpawel 1 |
| `minerals_gems` | 624 | Salniarwite, Frlozancite, Lanborlitte, Manelsite, Waunbsomite |
| `plants_flowers` | 634 | S Brierwork, Frlozenca, Brilybore, Dtelan Plant, Sarusberry |
| `perfumes` | 639 | Blantud Weelue, Ungto, Emaodcara, Pis Bluetr, Shramono |
| `metal_bands` | 663 | Rix Sadgrey, Kasvers, Kamand Argatis, Blirerow, Samon  |
| `video_games` | 753 | Rist, Skires of the Pars of Emalais, Blure ow tazon Showl, Oltiofy, Bonthalk |
| `greek_myth` | 755 | Taesthus, Lynala, Grissus, Mora, Tyrtomilion |
| `birds` | 863 | Rock Turk, Emiperin Hawk, Cap Torbler, Haucan Eagle-Wren, Townsend Penguin |
| `car_models` | 1218 | Supurba, Mimarin, GTa, Baxtrag, Boming |
| `world_cities` | 1691 | Pricons, Guiyk, Re Semf Baba, Towhasú, Eulllat |
| `pharma_drugs` | 2223 | Ribicolyd, Elicalprexen, Eclofen, Vozafrotine, Etimpefene |
| `english_words` | 8631 | tanza, complete, red, premiancy, lofder |

### Samples along the ladder

The qualitative side of section 1 — one domain, increasing data.

| dataset | train names | samples |
|---|---:|---|
| `motorcycle_brands` | 54 | BMclSndgrenanPHuertd, EaToicnranais, BHirlriw, Gamneo, IerFFoltak |
| `motorcycle_brands` | 130 | MMci, Sdgrerl, Piverto, EmTon, Erangi |
| `motorcycle_brands` | 263 | Ricu, Sdirer, Kasungto, Nangicg, Gacin |
| `car_manufacturers` | 135 | BMfo, Sdimenanaivertd, Janowchran, isuc, Srtr |
| `car_manufacturers` | 250 | BMxi, Sdimer, Kasverto, Nanowch, Latin |
| `car_manufacturers` | 502 | Rixsondgien, Kauverto, Nangich, Gatis, Binter |
| `spacecraft` | 230 | Surpead Sorer, Seravhoya, SAAn, Mengou, Ulalsa |
| `spacecraft` | 504 | Surpean, Soryuan, Fovo, Eatlinn, Meras |
| `car_models` | 218 | Swenega, Gra, Eiturta, Baxterl, BoCiiur |
| `car_models` | 400 | Suvtega, GTm, Eitura, Araxte, ElBiisiur |
| `car_models` | 700 | Suvuena, Gra, Eirche, Araxte, LaBo |
| `car_models` | 1035 | Supurga, Gra, Eitura, Araxtura, Bolinus |
| `english_words` | 500 | tundicky, traz, tred, pip, raaey |
| `english_words` | 1200 | tund, chawple, wirping, prease, alafder |
| `english_words` | 2500 | tands, comple, wired, pip, raney |

---

## Method

```
# library catalog (section 3), per dataset
python -m src.train --data data/<name>.txt --name ws11_<name> \
    --val-fraction 0.15 --patience 20 --epochs <ceiling>

# evaluation + single-checkpoint report
python -m src.evaluate --checkpoint checkpoints/ws11_<name>.pt \
    --num 200 --temperature 1.0 --seed 0 --report reports/<name>.md
```

Epoch ceilings: <200 names → 250; <800 → 180; <3000 → 120; else 60. Patience is a flat 20 at **every** size deliberately — a tighter early-stop rule on the large datasets would stop them earlier in their validation curve than the small ones and manufacture part of the very trend being measured. Everything else is stock `Config`: 2-layer 256-wide LSTM, embedding 32, dropout 0.2, Adam @ 3e-3, batch 32, constant LR, seed 1337. Best-epoch weights are restored before each checkpoint is written, so every number describes the model you would actually ship.

Both NLL columns come from `src/evaluate.py`'s `mean_char_nll`: token-weighted mean cross-entropy in nats/char with `model.eval()`, on the restored best-epoch weights. Token-weighting is what makes the two subtractable and the gap comparable across datasets with different name lengths. The `best val loss` column comes from the training loop instead, which averages per batch — so it is **not** directly comparable to the NLL columns, and appears only to show where each run stopped.

Sections 1 and 2 were driven by scripts outside the repo (this lane owns only `src/evaluate.py`, `tests/test_evaluate.py` and `reports/`); their raw per-run JSON is preserved under `reports/_bench/ladder/` and `reports/_bench/wave2proto/`.

Worked single-checkpoint `--report` output, in full: [`car_manufacturers.md`](car_manufacturers.md), [`birds.md`](birds.md).

### What this report does not establish

- **That any of this generalizes past a 2-layer 256-wide LSTM.** Every number here is one architecture at one size. WS-9's GRU and transformer runs are where that gets tested, and it is entirely possible the better conclusion is "this model is too big for these datasets" rather than "these datasets are too small". The ladder's flattening returns lean that way.
- **Run-to-run variance.** One seed per arm. `STATUS.md` records that three identical `--val-fraction 0.15` runs were not bit-reproducible before `Config.seed_init` was fixed, so differences below ~0.05 nats/char should be read as noise.
- **That the four grown datasets grew representatively.** The files are alphabetically sorted, so a grown file's added names cannot be separated from its original ones by position; truncating to the old line count would yield a biased A–M slice. The ladder arms are therefore size-matched *random subsamples of the current file*, not the historical file.
- **That the small datasets' held-out sets are large enough.** `motorcycle_brands` holds out 46 names; its held-out NLL is far noisier than `english_words`'s 1,294.
- **The top of the `english_words` ladder.** Arms at 5,000 and 7,337 training names were planned and cut for time — the box was shared with three other training lanes at load ~12 on 4 cores. That ladder therefore spans 500–2,500 names, not the full range, and it is the one ladder whose flattening is least well characterised.

Signed: Claude Code | Opus 5 | high
