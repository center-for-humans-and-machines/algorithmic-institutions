### frontier: PR 181 stimulus skip x joint-exodus GNN switch, lin_multinomial copula

| row | before | after | delta | band | seed_sd | in_seed_sd | legible | ungateable | target |
|---|---|---|---|---|---|---|---|---|---|
| CA | 0.8422 | 0.8417 | -0.0006 | <= 1 | 0.1878 | 0.0000 | False | True | False |
| CB | 0.8164 | 0.8161 | -0.0003 | <= 1 | 0.2153 | 0.0000 | False | True | False |
| CC | 0.8154 | 0.8151 | -0.0003 | <= 1 | 0.1333 | 0.0000 | False | True | False |
| CD | 0.7665 | 0.7659 | -0.0005 | <= 1 | 0.1878 | 0.0000 | False | True | False |
| CE | 0.9675 | 0.9675 | 0.0000 | <= 1 | 0.0581 | 0.0000 | False | False | False |
| CF | 0.8152 | 0.8144 | -0.0008 | <= 1 | 0.1413 | 0.0100 | False | True | False |
| CG | 1.8449 | 1.8461 | 0.0012 | 1-2 | 0.3014 | 0.0000 | False | True | True |
| SA | 0.7741 | 0.7741 | 0.0000 | <= 1 | 0.1618 | 0.0000 | False | True | False |
| SB | 1.0695 | 1.0695 | 0.0000 | 1-2 | 0.0456 | 0.0000 | False | True | False |
| SC | 1.8295 | 1.8295 | 0.0000 | 1-2 | 0.1355 | 0.0000 | False | False | False |
| PA | 0.6408 | 0.6413 | 0.0005 | <= 1 | 0.0404 | 0.0100 | False | False | False |
| PB | 0.8995 | 0.8999 | 0.0004 | <= 1 | 0.0231 | 0.0200 | False | False | False |
| PC | 0.9147 | 0.9146 | -0.0001 | <= 1 | 0.0359 | 0.0000 | False | False | False |
| PD | 0.6860 | 0.6804 | -0.0057 | <= 1 | 0.0593 | 0.1000 | False | False | False |
| RCA | 1.7761 | 1.7730 | -0.0031 | 1-2 | 0.1412 | 0.0200 | False | False | True |
| RCB | 1.0668 | 1.0596 | -0.0072 | 1-2 | 0.1425 | 0.0500 | False | False | True |
| RCC | 1.4983 | 1.4974 | -0.0009 | 1-2 | 0.1631 | 0.0100 | False | False | True |
| RCD | 1.3368 | 1.3372 | 0.0004 | 1-2 | 0.2698 | 0.0000 | False | False | True |
| RCE | 0.9474 | 0.9260 | -0.0215 | <= 1 | 0.1063 | 0.2000 | False | True | True |
| RSA | 1.1529 | 1.1529 | 0.0000 | 1-2 | 0.1555 | 0.0000 | False | True | True |
| RPA | 0.6426 | 0.6417 | -0.0009 | <= 1 | 0.0175 | 0.0500 | False | False | True |
| RPB | 0.7624 | 0.7613 | -0.0011 | <= 1 | 0.0283 | 0.0400 | False | False | True |
| mean | 1.0393 | 1.0375 | -0.0018 |  | 0.0473 | 0.0400 | False | False | False |
| rows <= 1 | 14.0000 | 14.0000 | 0.0000 |  | 3.1623 | 0.0000 | False | True | False |

recorded output diff: {'rows': 19200, 'punishment_changed': 14, 'punishment_changed_share': 0.000729, 'contribution_changed': 35, 'contribution_changed_share': 0.001823, 'group_id_changed': 0, 'group_id_changed_share': 0.0, 'mean_punishment_before': 1.8464, 'mean_punishment_after': 1.8417, 'mean_contribution_before': 9.6595, 'mean_contribution_after': 9.6585, 'punishment_cells_zeroed': 5}

| RCE slopes | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| human | +0.140 +- 0.018 (n 965) | +0.104 +- 0.024 (n 929) | -0.077 +- 0.035 (n 560) | -0.161 +- 0.079 (n 206) |
| before | +0.102 +- 0.014 (n 2045) | +0.019 +- 0.015 (n 1906) | +0.000 +- 0.022 (n 1364) | -0.085 +- 0.054 (n 474) |
| after | +0.103 +- 0.014 (n 2043) | +0.024 +- 0.015 (n 1904) | +0.000 +- 0.022 (n 1364) | -0.085 +- 0.054 (n 474) |

protected-row checks (raw): {'band_downgrade': False, 'sign_lost': [], 'magnitude_halved': [], 'signs_before': '+++-', 'signs_after': '+++-', 'change_in_se': {'0-4': 0.04, '5-9': 0.26, '10-14': 0.0, '15-19': 0.0}, 'change_in_seed_sd': {'0-4': 0.04, '5-9': 0.25, '10-14': 0.0, '15-19': 0.0}, 'magnitude_eroded': [], 'magnitude_halved_but_toward_human': []}

protected-row checks (amended): {'band_downgrade_raw': False, 'band_downgrade_amended': False, 'band_drop_in_seed_sd': 0.2, 'sign_lost_raw': [], 'sign_lost_amended': [], 'magnitude_halved_raw': [], 'magnitude_eroded_amended': [], 'magnitude_halved_but_toward_human': [], 'change_in_se': {'0-4': 0.04, '5-9': 0.26, '10-14': 0.0, '15-19': 0.0}, 'change_in_seed_sd': {'0-4': 0.04, '5-9': 0.25, '10-14': 0.0, '15-19': 0.0}, 'rce_ungateable_on_one_run': True, 'rce_seed_sd': 0.1063, 'band_slope_seed_sd': {'0-4': 0.0182, '5-9': 0.0223, '10-14': 0.0263, '15-19': 0.0564}}

verdict: targets=['CG', 'RCA', 'RCB', 'RCC', 'RCD', 'RCE', 'RSA', 'RPA', 'RPB'], target_before={'CG': 1.8449, 'RCA': 1.7761, 'RCB': 1.0668, 'RCC': 1.4983, 'RCD': 1.3368, 'RCE': 0.9474, 'RSA': 1.1529, 'RPA': 0.6426, 'RPB': 0.7624}, target_after={'CG': 1.8461, 'RCA': 1.773, 'RCB': 1.0596, 'RCC': 1.4974, 'RCD': 1.3372, 'RCE': 0.926, 'RSA': 1.1529, 'RPA': 0.6417, 'RPB': 0.7613}, target_in_seed_sd={'CG': 0.0, 'RCA': 0.02, 'RCB': 0.05, 'RCC': 0.01, 'RCD': 0.0, 'RCE': 0.2, 'RSA': 0.0, 'RPA': 0.05, 'RPB': 0.04}, target_legible=[], gate1_band_upgrades=[], gate1_ok=False, mean_before=1.0393424415811934, mean_after=1.0375059385558443, gate2_ceiling=1.1432766857393128, gate2_mean_ok=True, mean_move_in_seed_sd=0.04, rce_protected_ok=True, verdict=FAIL

### ref_lin: main gnn x gnn, lin_multinomial (timeout feature)

| row | before | after | delta | band | seed_sd | in_seed_sd | legible | ungateable | target |
|---|---|---|---|---|---|---|---|---|---|
| CA | 0.7121 | 0.7121 | 0.0000 | <= 1 | 0.1878 | 0.0000 | False | True | False |
| CB | 0.6753 | 0.6753 | 0.0000 | <= 1 | 0.2153 | 0.0000 | False | True | False |
| CC | 1.5177 | 1.5177 | 0.0000 | 1-2 | 0.1333 | 0.0000 | False | True | False |
| CD | 0.6048 | 0.6048 | 0.0000 | <= 1 | 0.1878 | 0.0000 | False | True | False |
| CE | 1.1713 | 1.1713 | 0.0000 | 1-2 | 0.0581 | 0.0000 | False | False | False |
| CF | 0.7576 | 0.7576 | 0.0000 | <= 1 | 0.1413 | 0.0000 | False | True | False |
| CG | 9.4422 | 9.4422 | 0.0000 | > 5 | 0.3014 | 0.0000 | False | True | True |
| SA | 0.6874 | 0.6874 | 0.0000 | <= 1 | 0.1618 | 0.0000 | False | True | False |
| SB | 0.8109 | 0.8109 | 0.0000 | <= 1 | 0.0456 | 0.0000 | False | True | False |
| SC | 2.7218 | 2.7218 | 0.0000 | 2-5 | 0.1355 | 0.0000 | False | False | False |
| PA | 0.6199 | 0.6195 | -0.0004 | <= 1 | 0.0404 | 0.0100 | False | False | False |
| PB | 0.8422 | 0.8423 | 0.0001 | <= 1 | 0.0231 | 0.0100 | False | False | False |
| PC | 0.8053 | 0.8054 | 0.0001 | <= 1 | 0.0359 | 0.0000 | False | False | False |
| PD | 2.6623 | 2.6615 | -0.0007 | 2-5 | 0.0593 | 0.0100 | False | False | False |
| RCA | 2.1753 | 2.1753 | 0.0000 | 2-5 | 0.1412 | 0.0000 | False | False | True |
| RCB | 0.9732 | 0.9713 | -0.0019 | <= 1 | 0.1425 | 0.0100 | False | False | True |
| RCC | 1.4705 | 1.4705 | 0.0000 | 1-2 | 0.1631 | 0.0000 | False | False | True |
| RCD | 2.9584 | 2.9584 | 0.0000 | 2-5 | 0.2698 | 0.0000 | False | False | True |
| RCE | 0.9412 | 0.9410 | -0.0001 | <= 1 | 0.1063 | 0.0000 | False | True | True |
| RSA | 0.8800 | 0.8800 | 0.0000 | <= 1 | 0.1555 | 0.0000 | False | True | True |
| RPA | 0.6414 | 0.6416 | 0.0001 | <= 1 | 0.0175 | 0.0100 | False | False | True |
| RPB | 0.7575 | 0.7575 | 0.0000 | <= 1 | 0.0283 | 0.0000 | False | False | True |
| mean | 1.5831 | 1.5830 | -0.0001 |  | 0.0473 | 0.0000 | False | False | False |
| rows <= 1 | 14.0000 | 14.0000 | 0.0000 |  | 3.1623 | 0.0000 | False | True | False |

recorded output diff: {'rows': 19200, 'punishment_changed': 1, 'punishment_changed_share': 5.2e-05, 'contribution_changed': 0, 'contribution_changed_share': 0.0, 'group_id_changed': 0, 'group_id_changed_share': 0.0, 'mean_punishment_before': 1.8703, 'mean_punishment_after': 1.8699, 'mean_contribution_before': 9.6076, 'mean_contribution_after': 9.6076, 'punishment_cells_zeroed': 1}

| RCE slopes | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| human | +0.140 +- 0.018 (n 965) | +0.104 +- 0.024 (n 929) | -0.077 +- 0.035 (n 560) | -0.161 +- 0.079 (n 206) |
| before | +0.065 +- 0.014 (n 2147) | +0.094 +- 0.015 (n 1725) | +0.015 +- 0.022 (n 1407) | -0.078 +- 0.060 (n 471) |
| after | +0.065 +- 0.014 (n 2147) | +0.094 +- 0.015 (n 1724) | +0.015 +- 0.022 (n 1407) | -0.078 +- 0.060 (n 471) |

protected-row checks (raw): {'band_downgrade': False, 'sign_lost': [], 'magnitude_halved': [], 'signs_before': '+++-', 'signs_after': '+++-', 'change_in_se': {'0-4': 0.0, '5-9': 0.01, '10-14': 0.0, '15-19': 0.0}, 'change_in_seed_sd': {'0-4': 0.0, '5-9': 0.01, '10-14': 0.0, '15-19': 0.0}, 'magnitude_eroded': [], 'magnitude_halved_but_toward_human': []}

protected-row checks (amended): {'band_downgrade_raw': False, 'band_downgrade_amended': False, 'band_drop_in_seed_sd': 0.0, 'sign_lost_raw': [], 'sign_lost_amended': [], 'magnitude_halved_raw': [], 'magnitude_eroded_amended': [], 'magnitude_halved_but_toward_human': [], 'change_in_se': {'0-4': 0.0, '5-9': 0.01, '10-14': 0.0, '15-19': 0.0}, 'change_in_seed_sd': {'0-4': 0.0, '5-9': 0.01, '10-14': 0.0, '15-19': 0.0}, 'rce_ungateable_on_one_run': True, 'rce_seed_sd': 0.1063, 'band_slope_seed_sd': {'0-4': 0.0182, '5-9': 0.0223, '10-14': 0.0263, '15-19': 0.0564}}

### ref_gnn: main gnn x gnn, gnn punisher (timeout feature)

| row | before | after | delta | band | seed_sd | in_seed_sd | legible | ungateable | target |
|---|---|---|---|---|---|---|---|---|---|
| CA | 0.7465 | 0.7353 | -0.0112 | <= 1 | 0.1878 | 0.0600 | False | True | False |
| CB | 0.7592 | 0.7523 | -0.0069 | <= 1 | 0.2153 | 0.0300 | False | True | False |
| CC | 1.5025 | 1.5301 | 0.0276 | 1-2 | 0.1333 | 0.2100 | False | True | False |
| CD | 0.7049 | 0.6976 | -0.0073 | <= 1 | 0.1878 | 0.0400 | False | True | False |
| CE | 1.2358 | 1.2989 | 0.0630 | 1-2 | 0.0581 | 1.0800 | True | False | False |
| CF | 0.9221 | 0.8815 | -0.0406 | <= 1 | 0.1413 | 0.2900 | False | True | False |
| CG | 9.5633 | 9.9068 | 0.3435 | > 5 | 0.3014 | 1.1400 | True | True | True |
| SA | 1.0466 | 0.8744 | -0.1722 | 1-2 -> <= 1 | 0.1618 | 1.0600 | True | True | False |
| SB | 0.9961 | 0.9353 | -0.0608 | <= 1 | 0.0456 | 1.3300 | True | True | False |
| SC | 2.5394 | 2.8168 | 0.2774 | 2-5 | 0.1355 | 2.0500 | True | False | False |
| PA | 2.0917 | 1.6989 | -0.3928 | 2-5 -> 1-2 | 0.0404 | 9.7200 | True | False | False |
| PB | 1.4959 | 1.2812 | -0.2147 | 1-2 | 0.0231 | 9.2900 | True | False | False |
| PC | 1.2573 | 1.0881 | -0.1692 | 1-2 | 0.0359 | 4.7100 | True | False | False |
| PD | 2.8678 | 2.9060 | 0.0382 | 2-5 | 0.0593 | 0.6400 | False | False | False |
| RCA | 2.3969 | 2.3627 | -0.0342 | 2-5 | 0.1412 | 0.2400 | False | False | True |
| RCB | 1.4271 | 1.3616 | -0.0655 | 1-2 | 0.1425 | 0.4600 | False | False | True |
| RCC | 1.0515 | 1.0687 | 0.0171 | 1-2 | 0.1631 | 0.1100 | False | False | True |
| RCD | 2.7921 | 2.8540 | 0.0620 | 2-5 | 0.2698 | 0.2300 | False | False | True |
| RCE | 1.0322 | 1.0309 | -0.0012 | 1-2 | 0.1063 | 0.0100 | False | True | True |
| RSA | 0.9445 | 0.9394 | -0.0051 | <= 1 | 0.1555 | 0.0300 | False | True | True |
| RPA | 1.0927 | 0.9657 | -0.1270 | 1-2 -> <= 1 | 0.0175 | 7.2600 | True | False | True |
| RPB | 2.0202 | 1.7290 | -0.2912 | 2-5 -> 1-2 | 0.0283 | 10.2900 | True | False | True |
| mean | 1.8403 | 1.8052 | -0.0350 |  | 0.0473 | 0.7400 | False | False | False |
| rows <= 1 | 6.0000 | 8.0000 | 2.0000 |  | 3.1623 | 0.6300 | False | True | False |

recorded output diff: {'rows': 19200, 'punishment_changed': 444, 'punishment_changed_share': 0.023125, 'contribution_changed': 505, 'contribution_changed_share': 0.026302, 'group_id_changed': 484, 'group_id_changed_share': 0.025208, 'mean_punishment_before': 2.4878, 'mean_punishment_after': 2.3576, 'mean_contribution_before': 9.0753, 'mean_contribution_after': 9.0877, 'punishment_cells_zeroed': 326}

| RCE slopes | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| human | +0.140 +- 0.018 (n 965) | +0.104 +- 0.024 (n 929) | -0.077 +- 0.035 (n 560) | -0.161 +- 0.079 (n 206) |
| before | +0.060 +- 0.012 (n 2862) | +0.052 +- 0.013 (n 2099) | +0.012 +- 0.019 (n 1259) | -0.012 +- 0.045 (n 399) |
| after | +0.062 +- 0.012 (n 2766) | +0.055 +- 0.013 (n 1972) | +0.024 +- 0.019 (n 1233) | -0.022 +- 0.046 (n 384) |

protected-row checks (raw): {'band_downgrade': False, 'sign_lost': [], 'magnitude_halved': [], 'signs_before': '+++-', 'signs_after': '+++-', 'change_in_se': {'0-4': 0.14, '5-9': 0.17, '10-14': 0.44, '15-19': 0.15}, 'change_in_seed_sd': {'0-4': 0.13, '5-9': 0.14, '10-14': 0.45, '15-19': 0.17}, 'magnitude_eroded': [], 'magnitude_halved_but_toward_human': []}

protected-row checks (amended): {'band_downgrade_raw': False, 'band_downgrade_amended': False, 'band_drop_in_seed_sd': 0.01, 'sign_lost_raw': [], 'sign_lost_amended': [], 'magnitude_halved_raw': [], 'magnitude_eroded_amended': [], 'magnitude_halved_but_toward_human': [], 'change_in_se': {'0-4': 0.14, '5-9': 0.17, '10-14': 0.44, '15-19': 0.15}, 'change_in_seed_sd': {'0-4': 0.13, '5-9': 0.14, '10-14': 0.45, '15-19': 0.17}, 'rce_ungateable_on_one_run': True, 'rce_seed_sd': 0.1063, 'band_slope_seed_sd': {'0-4': 0.0182, '5-9': 0.0223, '10-14': 0.0263, '15-19': 0.0564}}
