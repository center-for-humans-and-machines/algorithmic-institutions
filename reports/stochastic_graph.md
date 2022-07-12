```mermaid
flowchart TD

pCG ---> Gr
pCon ---> Gr
pPay ---> Gr

Gr ---> Con

Mech ---> Fac
Con ---> Mech

Fac ---> Pay
CG ---> Pay

Con ---> CG




pGr ---> pCon

pMech ---> pFac
pCon ---> pMech

pFac ---> pPay
pCG ---> pPay

pCon ---> pCG
```

```mermaid
flowchart TD

pCG ---> Gr
pCon ---> Gr
pPun ---> Gr

Gr ---> Con


Con ---> Man
pPun

Man ---> Pun



Con ---> Mech

Fac ---> Pay
CG ---> Pay

Con ---> CG




pGr ---> pCon

pMech ---> pFac
pCon ---> pMech

pFac ---> pPay
pCG ---> pPay

pCon ---> pCG
```
