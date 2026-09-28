### frontier: PR 181 stimulus skip x joint-exodus GNN switch, lin_multinomial copula

| row | before | after | delta | band | seed_sd | in_seed_sd | legible | ungateable | target |
|---|---|---|---|---|---|---|---|---|---|
| CA | 0.8588 | 0.8422 | -0.0165 | <= 1 | 0.1878 | 0.0900 | False | True | False |
| CB | 0.7850 | 0.8164 | 0.0314 | <= 1 | 0.2153 | 0.1500 | False | True | False |
| CC | 0.8387 | 0.8154 | -0.0233 | <= 1 | 0.1333 | 0.1700 | False | True | False |
| CD | 0.7939 | 0.7665 | -0.0274 | <= 1 | 0.1878 | 0.1500 | False | True | False |
| CE | 0.9880 | 0.9675 | -0.0205 | <= 1 | 0.0581 | 0.3500 | False | False | False |
| CF | 0.8414 | 0.8152 | -0.0262 | <= 1 | 0.1413 | 0.1900 | False | True | False |
| CG | 1.2975 | 1.8449 | 0.5475 | 1-2 | 0.3014 | 1.8200 | True | True | True |
| SA | 0.8744 | 0.7741 | -0.1003 | <= 1 | 0.1618 | 0.6200 | False | True | True |
| SB | 0.9598 | 1.0695 | 0.1096 | <= 1 -> 1-2 | 0.0456 | 2.4000 | True | True | True |
| SC | 1.5278 | 1.8295 | 0.3017 | 1-2 | 0.1355 | 2.2300 | True | False | True |
| PA | 0.6375 | 0.6408 | 0.0033 | <= 1 | 0.0404 | 0.0800 | False | False | False |
| PB | 0.8578 | 0.8995 | 0.0417 | <= 1 | 0.0231 | 1.8000 | True | False | False |
| PC | 0.8727 | 0.9147 | 0.0420 | <= 1 | 0.0359 | 1.1700 | True | False | False |
| PD | 0.8854 | 0.6860 | -0.1993 | <= 1 | 0.0593 | 3.3600 | True | False | False |
| RCA | 1.6715 | 1.7761 | 0.1046 | 1-2 | 0.1412 | 0.7400 | False | False | False |
| RCB | 1.1797 | 1.0668 | -0.1129 | 1-2 | 0.1425 | 0.7900 | False | False | False |
| RCC | 1.0769 | 1.4983 | 0.4214 | 1-2 | 0.1631 | 2.5800 | True | False | False |
| RCD | 1.1313 | 1.3368 | 0.2055 | 1-2 | 0.2698 | 0.7600 | False | False | False |
| RCE | 0.8719 | 0.9474 | 0.0755 | <= 1 | 0.1063 | 0.7100 | False | True | False |
| RSA | 1.2595 | 1.1529 | -0.1066 | 1-2 | 0.1555 | 0.6900 | False | True | False |
| RPA | 0.6375 | 0.6426 | 0.0051 | <= 1 | 0.0175 | 0.2900 | False | False | False |
| RPB | 0.7653 | 0.7624 | -0.0029 | <= 1 | 0.0283 | 0.1000 | False | False | False |
| mean | 0.9824 | 1.0393 | 0.0570 |  | 0.0473 | 1.2000 | True | False | False |
| rows <= 1 | 15.0000 | 14.0000 | -1.0000 |  | 3.1623 | 0.3200 | False | True | False |

| RCE slopes | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| human | +0.140 +- 0.018 (n 965) | +0.104 +- 0.024 (n 929) | -0.077 +- 0.035 (n 560) | -0.161 +- 0.079 (n 206) |
| before | +0.092 +- 0.015 (n 2002) | +0.034 +- 0.014 (n 1919) | -0.048 +- 0.021 (n 1394) | -0.097 +- 0.055 (n 436) |
| after | +0.102 +- 0.014 (n 2045) | +0.019 +- 0.015 (n 1906) | +0.000 +- 0.022 (n 1364) | -0.085 +- 0.054 (n 474) |

protected-row checks: {'band_downgrade': False, 'sign_lost': ['10-14'], 'magnitude_halved': ['10-14'], 'signs_before': '++--', 'signs_after': '+++-', 'change_in_se': {'0-4': 0.5, '5-9': 0.72, '10-14': 1.57, '15-19': 0.16}, 'change_in_seed_sd': {'0-4': 0.55, '5-9': 0.67, '10-14': 1.83, '15-19': 0.21}, 'magnitude_eroded': ['10-14'], 'magnitude_halved_but_toward_human': []}

verdict: targets=['SA', 'SB', 'SC', 'CG'], target_before={'SA': 0.8744, 'SB': 0.9598, 'SC': 1.5278, 'CG': 1.2975}, target_after={'SA': 0.7741, 'SB': 1.0695, 'SC': 1.8295, 'CG': 1.8449}, target_in_seed_sd={'SA': 0.62, 'SB': 2.4, 'SC': 2.23, 'CG': 1.82}, gate1_band_upgrades=[], gate1_ok=False, mean_before=0.9823697087542729, mean_after=1.0393424415811934, gate2_ceiling=1.0806066796297003, gate2_mean_ok=True, mean_move_in_seed_sd=1.2, rce_protected_ok=False, rce_ungateable_on_one_run=True, verdict=FAIL

### ref_lin: main gnn x gnn, lin_multinomial (no copula)

| row | before | after | delta | band | seed_sd | in_seed_sd | legible | ungateable | target |
|---|---|---|---|---|---|---|---|---|---|
| CA | 0.9794 | 0.7121 | -0.2673 | <= 1 | 0.1878 | 1.4200 | True | True | False |
| CB | 0.8617 | 0.6753 | -0.1864 | <= 1 | 0.2153 | 0.8700 | False | True | False |
| CC | 1.6900 | 1.5177 | -0.1724 | 1-2 | 0.1333 | 1.2900 | True | True | False |
| CD | 0.8511 | 0.6048 | -0.2463 | <= 1 | 0.1878 | 1.3100 | True | True | False |
| CE | 1.2065 | 1.1713 | -0.0351 | 1-2 | 0.0581 | 0.6000 | False | False | False |
| CF | 0.8617 | 0.7576 | -0.1041 | <= 1 | 0.1413 | 0.7400 | False | True | False |
| CG | 9.0888 | 9.4422 | 0.3534 | > 5 | 0.3014 | 1.1700 | True | True | True |
| SA | 0.7361 | 0.6874 | -0.0487 | <= 1 | 0.1618 | 0.3000 | False | True | True |
| SB | 0.7853 | 0.8109 | 0.0256 | <= 1 | 0.0456 | 0.5600 | False | True | True |
| SC | 3.1142 | 2.7218 | -0.3923 | 2-5 | 0.1355 | 2.9000 | True | False | True |
| PA | 0.5800 | 0.6199 | 0.0399 | <= 1 | 0.0404 | 0.9900 | False | False | False |
| PB | 0.8218 | 0.8422 | 0.0204 | <= 1 | 0.0231 | 0.8800 | False | False | False |
| PC | 0.7884 | 0.8053 | 0.0169 | <= 1 | 0.0359 | 0.4700 | False | False | False |
| PD | 2.7419 | 2.6623 | -0.0796 | 2-5 | 0.0593 | 1.3400 | True | False | False |
| RCA | 1.9871 | 2.1753 | 0.1882 | 1-2 -> 2-5 | 0.1412 | 1.3300 | True | False | False |
| RCB | 0.9679 | 0.9732 | 0.0053 | <= 1 | 0.1425 | 0.0400 | False | False | False |
| RCC | 1.5156 | 1.4705 | -0.0450 | 1-2 | 0.1631 | 0.2800 | False | False | False |
| RCD | 3.0557 | 2.9584 | -0.0973 | 2-5 | 0.2698 | 0.3600 | False | False | False |
| RCE | 1.0408 | 0.9412 | -0.0996 | 1-2 -> <= 1 | 0.1063 | 0.9400 | False | True | False |
| RSA | 0.9336 | 0.8800 | -0.0536 | <= 1 | 0.1555 | 0.3400 | False | True | False |
| RPA | 0.6571 | 0.6414 | -0.0156 | <= 1 | 0.0175 | 0.8900 | False | False | False |
| RPB | 0.7157 | 0.7575 | 0.0418 | <= 1 | 0.0283 | 1.4800 | True | False | False |
| mean | 1.6355 | 1.5831 | -0.0524 |  | 0.0473 | 1.1100 | True | False | False |
| rows <= 1 | 13.0000 | 14.0000 | 1.0000 |  | 3.1623 | 0.3200 | False | True | False |

| RCE slopes | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| human | +0.140 +- 0.018 (n 965) | +0.104 +- 0.024 (n 929) | -0.077 +- 0.035 (n 560) | -0.161 +- 0.079 (n 206) |
| before | +0.054 +- 0.016 (n 1781) | +0.106 +- 0.015 (n 1801) | +0.045 +- 0.021 (n 1483) | -0.064 +- 0.061 (n 490) |
| after | +0.065 +- 0.014 (n 2147) | +0.094 +- 0.015 (n 1725) | +0.015 +- 0.022 (n 1407) | -0.078 +- 0.060 (n 471) |

protected-row checks: {'band_downgrade': False, 'sign_lost': [], 'magnitude_halved': ['10-14'], 'signs_before': '+++-', 'signs_after': '+++-', 'change_in_se': {'0-4': 0.51, '5-9': 0.58, '10-14': 0.98, '15-19': 0.16}, 'change_in_seed_sd': {'0-4': 0.59, '5-9': 0.54, '10-14': 1.15, '15-19': 0.24}, 'magnitude_eroded': [], 'magnitude_halved_but_toward_human': ['10-14']}

### ref_gnn: main gnn x gnn, gnn punisher

| row | before | after | delta | band | seed_sd | in_seed_sd | legible | ungateable | target |
|---|---|---|---|---|---|---|---|---|---|
| CA | 0.7768 | 0.7465 | -0.0303 | <= 1 | 0.1878 | 0.1600 | False | True | False |
| CB | 0.6821 | 0.7592 | 0.0771 | <= 1 | 0.2153 | 0.3600 | False | True | False |
| CC | 1.5452 | 1.5025 | -0.0428 | 1-2 | 0.1333 | 0.3200 | False | True | False |
| CD | 0.6579 | 0.7049 | 0.0470 | <= 1 | 0.1878 | 0.2500 | False | True | False |
| CE | 1.3696 | 1.2358 | -0.1337 | 1-2 | 0.0581 | 2.3000 | True | False | False |
| CF | 0.8081 | 0.9221 | 0.1140 | <= 1 | 0.1413 | 0.8100 | False | True | False |
| CG | 9.2739 | 9.5633 | 0.2894 | > 5 | 0.3014 | 0.9600 | False | True | True |
| SA | 0.9473 | 1.0466 | 0.0993 | <= 1 -> 1-2 | 0.1618 | 0.6100 | False | True | True |
| SB | 0.9147 | 0.9961 | 0.0814 | <= 1 | 0.0456 | 1.7800 | True | True | True |
| SC | 2.9539 | 2.5394 | -0.4145 | 2-5 | 0.1355 | 3.0600 | True | False | True |
| PA | 1.9776 | 2.0917 | 0.1141 | 1-2 -> 2-5 | 0.0404 | 2.8200 | True | False | False |
| PB | 1.4317 | 1.4959 | 0.0642 | 1-2 | 0.0231 | 2.7800 | True | False | False |
| PC | 1.2123 | 1.2573 | 0.0450 | 1-2 | 0.0359 | 1.2500 | True | False | False |
| PD | 3.2712 | 2.8678 | -0.4034 | 2-5 | 0.0593 | 6.8000 | True | False | False |
| RCA | 2.2153 | 2.3969 | 0.1816 | 2-5 | 0.1412 | 1.2900 | True | False | False |
| RCB | 1.3461 | 1.4271 | 0.0810 | 1-2 | 0.1425 | 0.5700 | False | False | False |
| RCC | 1.1100 | 1.0515 | -0.0584 | 1-2 | 0.1631 | 0.3600 | False | False | False |
| RCD | 2.6329 | 2.7921 | 0.1591 | 2-5 | 0.2698 | 0.5900 | False | False | False |
| RCE | 0.9653 | 1.0322 | 0.0669 | <= 1 -> 1-2 | 0.1063 | 0.6300 | False | True | False |
| RSA | 0.9504 | 0.9445 | -0.0059 | <= 1 | 0.1555 | 0.0400 | False | True | False |
| RPA | 1.1280 | 1.0927 | -0.0353 | 1-2 | 0.0175 | 2.0200 | True | False | False |
| RPB | 1.9325 | 2.0202 | 0.0877 | 1-2 -> 2-5 | 0.0283 | 3.1000 | True | False | False |
| mean | 1.8229 | 1.8403 | 0.0174 |  | 0.0473 | 0.3700 | False | False | False |
| rows <= 1 | 8.0000 | 6.0000 | -2.0000 |  | 3.1623 | 0.6300 | False | True | False |

| RCE slopes | 0-4 | 5-9 | 10-14 | 15-19 |
|---|---|---|---|---|
| human | +0.140 +- 0.018 (n 965) | +0.104 +- 0.024 (n 929) | -0.077 +- 0.035 (n 560) | -0.161 +- 0.079 (n 206) |
| before | +0.066 +- 0.013 (n 2505) | +0.055 +- 0.013 (n 2214) | -0.004 +- 0.018 (n 1398) | -0.007 +- 0.041 (n 427) |
| after | +0.060 +- 0.012 (n 2862) | +0.052 +- 0.013 (n 2099) | +0.012 +- 0.019 (n 1259) | -0.012 +- 0.045 (n 399) |

protected-row checks: {'band_downgrade': True, 'sign_lost': ['10-14'], 'magnitude_halved': [], 'signs_before': '++--', 'signs_after': '+++-', 'change_in_se': {'0-4': 0.35, '5-9': 0.15, '10-14': 0.64, '15-19': 0.09}, 'change_in_seed_sd': {'0-4': 0.34, '5-9': 0.13, '10-14': 0.63, '15-19': 0.1}, 'magnitude_eroded': [], 'magnitude_halved_but_toward_human': []}
