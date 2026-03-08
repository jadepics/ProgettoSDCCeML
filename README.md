TODO
in ordine:

alberi per training NON distribuito

client che fa partire il training


Leggere:
introdotto docker per cointeiner dei diversi worker, master unico, worker arbitrari
struttura distribuita instanziata dataset caricato, bisogna ora creare la base ML per iniziare il training
e poi raffinare la parte distribuita


Comandi:

Fare benchmark, cambiando il numero dopo worker si aprono diversi worker, ai worker è già stata applicata una
scalabilità basata su id
docker compose down -v
docker compose up --build --scale worker=1