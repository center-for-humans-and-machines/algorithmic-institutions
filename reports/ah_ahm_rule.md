# Artificial Humans

We train 'artificial humans' that are predicting the next contribution based on historic contributions and punishments of all participants in a group. 

## Neural architecture

```mermaid
flowchart TD

G[Global] --> N[Node Model]
G[Global] --> E
I1[Node A] --> E[Edge Model]
I2[Node B] --> E[Edge Model]
E --> M1[Average over all edges of A] --> N
I1 --> N

N --> GRU1[Temporal Model] --> F3[Node Output A]


style F3 stroke-width:0px
style G stroke-width:0px
style I1 stroke-width:0px
style I2 stroke-width:0px
```

## Inputs

|   | Global  | Node  |
|---|---|---|
| Group Member | previous common good  <br /> round number | previous contributions <br /> previous punishments <br />  previous valid contribution |
| Manager | previous common good  <br /> round number | contributions <br /> previous punishments <br />  previous valid punishment |



