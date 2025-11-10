# Spaceship Titanic - Kaggle Challenge

In questo Readme verranno appuntate tutte le scelte tecniche prese al livello di scrittura del codice per il clean coding e per la leggibilità.

src:
|data_analysis.py
    ## Scelte implementative di codice

    ### Type Hinting
        Ho scelto di utilizzare il Type Hinting (Suggerimenti di Tipo) per migliorare la leggibilità, la manutenibilità e la possibilità di effettuare analisi statica del codice.

        Ad esempio nella funzione
        """
        def plot_age_distribution(train: pd.DataFrame, feature: str = "Age", target_col: str = "Transported") -> None:

        """

        Abbiamo:
        - 1: """ train: pd.DataFrame """
            - Nome del parametro: """train"""
            - Suggerimento di Tipo """: pd.DataFrame"""
                - Questo indica che ci si aspetta che l'argomento passato per """train""" sia un oggetto di tipo """DataFrame""" della libreria """pandas""" (importata con alias """pd""").
            - è il set di dati che la funzione deve analizzare.
        - 2. feature: str = "Age"
            - Nome del Parametro: feature
            - Suggerimento di Tipo: : str
                - Questo indica che ci si aspetta che l'argomento passato per feature sia una stringa (str). La stringa rappresenta il nome della colonna nel DataFrame che contiene i dati che si desidera tracciare (nel tuo caso, l'età).
            -Valore di Default: = "Age"
                - Significa che se non viene specificato un valore per feature durante la chiamata della funzione, verrà automaticamente utilizzato il valore "Age".
        - 4. -> None
            - Suggerimento di Tipo per il Valore di Ritorno: -> None
                - Questo indica che la funzione non restituisce esplicitamente alcun valore (cioè, restituisce None).
                - Nel contesto di una funzione di plot, questo è tipico perché il suo scopo è visualizzare un grafico (un effetto collaterale) e non calcolare o restituire un nuovo oggetto dati.

    ### Definizione dei percorsi

        """
        BASE_DIR = Path(__file__).resolve().parents[1]
        DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "train.csv"
        DEFAULT_TEST_PATH = BASE_DIR / "data" / "test.csv"
        """

        I percorsi vengono definiti in questo modo per garantire la portabilità e l'affidabilità del codice, indipendentemente dal sistema operativo su cui viene eseguito (Windows, macOS, Linux) e dalla directory di lavoro corrente. Questa metodologia sfrutta il modulo pathlib (e in particolare la classe Path).

        1. 
        """
        BASE_DIR = Path(__file__).resolve().parents[1]
        """
        - Questa riga complessa ha lo scopo di individuare in modo assoluto e robusto la cartella radice (root) del progetto, indipendentemente da dove viene eseguito lo script.

        """
        Path(__file__):
        """

            - """ __file__ """ è una variabile speciale di Python che contiene il percorso del file Python corrente (quello in cui si trova il codice che stai guardando).

            - """ Path() """ lo converte in un oggetto Path di pathlib, che permette di utilizzare i metodi orientati agli oggetti per la manipolazione dei percorsi.

        """
        .resolve():
        """
            - Trasforma il percorso in un percorso assoluto (completo, che parte dalla radice del file system, es. /home/user/... o C:\Users\...).
        
        """
        .parents[1] :
        """
            - L'attributo .parents è una sequenza di tutti i directory genitori (superiori) del percorso.

            parents[0] è la cartella immediatamente superiore (il genitore del file stesso).

            parents[1] è la cartella due livelli sopra il file corrente (il nonno).

            Perché [1]? Viene utilizzato per risalire dalla sottocartella (src) al percorso principale del progetto (progetto) in questa specific configurazione.
    
    ### Creeazione istogramma
        La funzione che stai utilizzando è sns.histplot(), che crea un istogramma per visualizzare la distribuzione di una variabile numerica.
        Ecco la scomposizione dei parametri:

            1. """ data=train """
                Significato: Specifica il DataFrame che contiene i dati da visualizzare.

                Ruolo: Indica a Seaborn di usare il DataFrame chiamato train per trovare le colonne specificate successivamente.

            2. """ x=feature """
                Significato: Definisce la variabile da visualizzare sull'asse delle ascisse (X).

                Ruolo: Questa è la colonna numerica di cui vuoi vedere la distribuzione. Nel tuo caso, se hai usato i valori predefiniti, sarà la colonna "Age" (Età).

            3. """ hue=target_col if target_col in train.columns else None """
                Significato: Aggiunge una dimensione di raggruppamento (colore) al grafico.

                Ruolo: Questo è un operatore condizionale (ternario) di Python che fa quanto segue:

                Se la colonna specificata in target_col (ad esempio "Transported") esiste (in train.columns) nel DataFrame train, allora usa quella colonna per colorare le barre dell'istogramma (dividendo la distribuzione per la categoria target).

                Altrimenti (else), imposta hue=None, creando un singolo istogramma senza divisioni per colore.

                Questo garantisce che la funzione non fallisca se la colonna target_col non è presente nel set di dati.

            4. """ binwidth=1 """
                Significato: Imposta la larghezza di ogni barra (bin) dell'istogramma.

                Ruolo: Specificando binwidth=1, stai chiedendo che ogni barra rappresenti un intervallo di ampiezza 1. Nel contesto dell'età, significa che le barre raggrupperanno gli individui di 0 anni, 1 anno, 2 anni, ecc. (assumendo che l'età sia misurata in anni interi).

            5. """ kde=True """
                Significato: Abilita la visualizzazione della stima della densità del kernel (Kernel Density Estimate).

                Ruolo: Disegna una linea morbida e continua sopra le barre dell'istogramma. Questa linea è una stima continua della distribuzione dei dati. È molto utile per visualizzare la forma generale della distribuzione e per confrontare meglio le distribuzioni dei diversi gruppi (definiti da hue). 
    
    ### Altre sclete implementative 

        - Riga xx
            """
            valid_features = [feature for feature in features if feature in train.columns]
            """ 

            Crea una lista di feature valide presenti nel DataFrame
            Itera su ogni elemento feature della lista features.
            Condizione if feature in train.columns: tiene solo le feature che esistono davvero tra le colonne del DataFrame train.
            Risultato: valid_features contiene le colonne richieste e presenti, nell’ordine originale; eventuali duplicati in features restano duplicati.
            Perché è utile:
            Evita errori quando mancano alcune colonne.
            Mantiene l’ordine dato dall’utente (importante per l’ordine dei grafici).

    
    