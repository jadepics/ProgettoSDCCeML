# SDCC/ML Client

Il client è il punto di ingresso esterno del sistema distribuito.

Il client non esegue training, non comunica con i worker e non accede direttamente agli artifact condivisi. Il suo compito è costruire una richiesta e inviarla al master tramite gRPC. Sarà poi il master a decidere come validare la richiesta, pianificare il job e distribuire il lavoro ai worker.

## Responsabilità del client

- leggere la configurazione dei master;
- costruire richieste gRPC verso il master;
- inviare richieste di training;
- mostrare all'utente la risposta ricevuta dal master.

## Cosa il client non deve fare

- non deve importare codice di `masterPackage`;
- non deve importare codice di `worker`;
- non deve leggere o modificare direttamente EFS`;
- non deve salvare artifact di training;
- non deve decidere la distribuzione del lavoro.

## Configurazione

Copiare il file di esempio:

```bash
cp client/.env.client.example client/.env.client