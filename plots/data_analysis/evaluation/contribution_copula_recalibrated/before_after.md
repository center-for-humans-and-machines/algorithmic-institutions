### frontier: PR 181 stimulus skip x joint-exodus GNN switch, lin_multinomial copula

before = `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_simtimeout`, after = `23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch_copularecal`, run `lin_multinomial_copula_self`

| row | before | after | delta | band | seed_sd | in_seed_sd | legible | ungateable | target |
|---|---|---|---|---|---|---|---|---|---|
| CA | 0.8422 | 0.8422 | 0.0000 | <= 1 | 0.1878 | 0.0000 | False | True | False |
| CB | 0.8164 | 0.8164 | 0.0000 | <= 1 | 0.2153 | 0.0000 | False | True | False |
| CC | 0.8154 | 0.8154 | 0.0000 | <= 1 | 0.1333 | 0.0000 | False | True | False |
| CD | 0.7665 | 0.7665 | 0.0000 | <= 1 | 0.1878 | 0.0000 | False | True | False |
| CE | 0.9675 | 0.9675 | 0.0000 | <= 1 | 0.0581 | 0.0000 | False | False | False |
| CF | 0.8152 | 0.8152 | 0.0000 | <= 1 | 0.1413 | 0.0000 | False | True | False |
| CG | 1.8449 | 1.8449 | 0.0000 | 1-2 | 0.3014 | 0.0000 | False | True | True |
| SA | 0.7741 | 0.7741 | 0.0000 | <= 1 | 0.1618 | 0.0000 | False | True | False |
| SB | 1.0695 | 1.0695 | 0.0000 | 1-2 | 0.0456 | 0.0000 | False | True | False |
| SC | 1.8295 | 1.8295 | 0.0000 | 1-2 | 0.1355 | 0.0000 | False | False | False |
| PA | 0.6408 | 0.6408 | 0.0000 | <= 1 | 0.0404 | 0.0000 | False | False | False |
| PB | 0.8995 | 0.8995 | 0.0000 | <= 1 | 0.0231 | 0.0000 | False | False | False |
| PC | 0.9147 | 0.9147 | 0.0000 | <= 1 | 0.0359 | 0.0000 | False | False | False |
| PD | 0.6860 | 0.6860 | 0.0000 | <= 1 | 0.0593 | 0.0000 | False | False | False |
| RCA | 1.7761 | 1.7761 | 0.0000 | 1-2 | 0.1412 | 0.0000 | False | False | False |
| RCB | 1.0668 | 1.0668 | 0.0000 | 1-2 | 0.1425 | 0.0000 | False | False | False |
| RCC | 1.4983 | 1.4983 | 0.0000 | 1-2 | 0.1631 | 0.0000 | False | False | False |
| RCD | 1.3368 | 1.3368 | 0.0000 | 1-2 | 0.2698 | 0.0000 | False | False | False |
| RCE | 0.9474 | 0.9474 | 0.0000 | <= 1 | 0.1063 | 0.0000 | False | True | False |
| RSA | 1.1529 | 1.1529 | 0.0000 | 1-2 | 0.1555 | 0.0000 | False | True | False |
| RPA | 0.6426 | 0.6426 | 0.0000 | <= 1 | 0.0175 | 0.0000 | False | False | False |
| RPB | 0.7624 | 0.7624 | 0.0000 | <= 1 | 0.0283 | 0.0000 | False | False | False |
| mean | 1.0393 | 1.0393 | 0.0000 |  | 0.0473 | 0.0000 | False | False | False |
| rows <= 1 | 14.0000 | 14.0000 | 0.0000 |  | 3.1623 | 0.0000 | False | True | False |

| RCE slopes | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| human | +0.140 +- 0.018 (n 965) | +0.104 +- 0.024 (n 929) | -0.077 +- 0.035 (n 560) | -0.161 +- 0.079 (n 206) |
| before | +0.102 +- 0.014 (n 2045) | +0.019 +- 0.015 (n 1906) | +0.000 +- 0.022 (n 1364) | -0.085 +- 0.054 (n 474) |
| after | +0.102 +- 0.014 (n 2045) | +0.019 +- 0.015 (n 1906) | +0.000 +- 0.022 (n 1364) | -0.085 +- 0.054 (n 474) |

protected-row checks (amended rule): band_downgrade=False, sign_lost=[], magnitude_halved=[], signs_before=+++-, signs_after=+++-, change_in_se={'0-4': 0.0, '5-9': 0.0, '10-14': 0.0, '15-19': 0.0}, change_in_seed_sd={'0-4': 0.0, '5-9': 0.0, '10-14': 0.0, '15-19': 0.0}, magnitude_eroded=[], magnitude_halved_but_toward_human=[], band_downgrade_raw=False, sign_lost_raw=[], magnitude_eroded_raw=[], fires=False

verdict: targets=['CG'], target_before={'CG': 1.8449}, target_after={'CG': 1.8449}, target_in_seed_sd={'CG': 0.0}, gate1_band_upgrades=[], gate1_upgrades_clearing_the_floor=[], gate1_ok=False, watch={'SC': (1.8295, 1.8295), 'RCC': (1.4983, 1.4983)}, mean_before=1.0393, mean_after=1.0393, gate2_ceiling=1.1433, gate2_mean_ok=True, mean_move_in_seed_sd=0.0, rce_protected_ok=True, rce_ungateable_on_one_run=True, verdict=FAIL
