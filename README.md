# Analiza opinii o Premier League z mediów społecznościowych

### 📝 Opis projektu
Projekt zajmuje się analizą sentymentu oraz badaniem najczęściej pojawiających się słów we wpisach na platformie Twitter dotyczących ligi piłkarskiej Premier League (#premierleague). Analiza obejmuje 12 000 tweetów z lat 2019–2023.

### ⚙️ Etapy przetwarzania (Preprocessing)
1.  **Tłumaczenie**: Wszystkie tweety zostały przetłumaczone na język angielski przy użyciu biblioteki `googletrans`.
2.  **Oczyszczanie tekstu**:
    * Zamiana na małe litery.
    * Usunięcie nazw użytkowników (@user) oraz znaków specjalnych.
    * Tokenizacja i usuwanie stop-words (w tym słów specyficznych dla kontekstu, np. "premier", "league").
    * Lematyzacja.

### 📊 Analiza i wizualizacja
* **Word Cloud**: Wizualizacja najpopularniejszych haseł.
* **Top 10 Words**: Wykresy słupkowe częstości występowania słów.
* **K-Means Clustering**: Grupowanie wpisów w tematyczne klastry.

### 🛠️ Wykorzystane technologie
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-C0C0C0?style=for-the-badge&logo=python&logoColor=3776AB)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
