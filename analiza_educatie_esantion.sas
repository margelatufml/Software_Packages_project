\
*---------------------------------------------------------------------------*;
* Script SAS Exemplu pentru Analiza Datelor Educaționale (Eșantion)         *;
*---------------------------------------------------------------------------*;

* Titlu general pentru output-ul SAS *;
TITLE 'Analiza Eșantion Date Educaționale';

*------------------------------------------------------;
* 1. Crearea unui set de date SAS cu date eșantion     *;
*    similar structurii datelor educaționale.          *;
*------------------------------------------------------;
DATA educatie_esantion;
    * Definirea lungimii pentru variabilele caracter *;
    LENGTH Regiune $ 20 Nivel_Educatie $ 50;
    
    * Specificarea variabilelor de intrare *;
    INPUT Regiune $ An Nivel_Educatie $ Valoare;
    
    * Linii de date (CARDS/DATALINES) *;
    CARDS;
Bucuresti       2020 Invatamant_primar      1500
Cluj            2020 Invatamant_primar      1200
Bucuresti       2021 Invatamant_primar      1550
Cluj            2021 Invatamant_primar      1250
Iasi            2020 Invatamant_secundar    900
Bucuresti       2020 Invatamant_secundar    1100
Cluj            2021 Invatamant_superior   2100
Bucuresti       2021 Invatamant_superior   2500
Iasi            2021 Invatamant_primar      750
Timis           2020 Invatamant_superior   1800
;
RUN;

*------------------------------------------------------;
* 2. Afișarea setului de date creat (PROC PRINT)       *;
*------------------------------------------------------;
PROC PRINT DATA=educatie_esantion;
    TITLE2 '2. Listarea Setului de Date educatie_esantion';
RUN;

*------------------------------------------------------;
* 3. Calcularea statisticilor descriptive (PROC MEANS) *;
*    pentru variabila 'Valoare', grupate.              *;
*------------------------------------------------------;
PROC MEANS DATA=educatie_esantion MEAN MEDIAN SUM MIN MAX N NMISS;
    CLASS Regiune Nivel_Educatie; /* Variabile de grupare */
    VAR Valoare;                  /* Variabila de analizat */
    TITLE2 '3. Statistici Descriptive pentru Valoare pe Regiune si Nivel Educație';
RUN;

*------------------------------------------------------;
* 4. Calcularea frecvențelor (PROC FREQ)               *;
*    pentru variabila 'Regiune'.                       *;
*------------------------------------------------------;
PROC FREQ DATA=educatie_esantion;
    TABLES Regiune / NOCUM NOPERCENT; /* Afișează doar frecvența */
    TITLE2 '4. Frecvența Înregistrărilor pe Regiune';
RUN;

*------------------------------------------------------;
* 5. Exemplu simplu de grafic cu SGPLOT (dacă SAS/GRAPH e disponibil) *;
*    Histograma pentru 'Valoare'.                       *;
*------------------------------------------------------;
ODS GRAPHICS ON; /* Activați ODS Graphics pentru SGPLOT */

PROC SGPLOT DATA=educatie_esantion;
    HISTOGRAM Valoare / BINWIDTH=200;
    DENSITY Valoare / TYPE=NORMAL; /* Suprapune curba normală */
    TITLE2 '5. Histograma pentru Variabila Valoare';
    XAXIS LABEL='Valoare';
    YAXIS LABEL='Frecvență';
RUN;

ODS GRAPHICS OFF; /* Dezactivați ODS Graphics */

* Curățare titlu pentru a nu afecta alte rulări *;
TITLE; 