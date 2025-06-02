import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dateutil
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import geopandas as gpd
import statsmodels.api as sm
from utils import load_data

# Attempt to import saspy. If not configured, it might still import but fail at runtime.
SAS_AVAILABLE = False
try:
    import saspy
    SAS_AVAILABLE = True
except ImportError:
    print("saspy library not found. SAS functionality will be limited.")
except Exception as e:
    # This might catch saspy configuration errors on import on some systems, though often they appear at SASsession()
    print(f"Error importing saspy (potentially configuration issue): {e}. SAS functionality will be limited.") 

st.title("Analiza Statistică a Datelor Educaționale din România")
st.sidebar.header("Meniu")

# Added sections for new analyses
section_options = ["Vizualizare Date", "Filtrare & Analiză", "Prelucrare & Grafice", "Codificare Date", "Scalare Date", "Outlieri", "Clusterizare (Scikit-learn)", "Regresie Logistică (Scikit-learn)", "Regresie Multiplă (Statsmodels)", "SAS Runner", "Demonstrare Facilități SAS"]
if SAS_AVAILABLE:
    section_options.append("SAS Runner")
else:
    section_options.append("SAS Runner (saspy indisponibil)")

section = st.sidebar.radio("Selectați o secțiune:", section_options)

if "data" not in st.session_state:
    try:
        data = load_data()
        st.session_state["data"] = data
    except Exception as e:
        st.error(f"Eroare la încărcarea datelor din setDate.xlsx: {e}")
        st.stop()

if section == "Vizualizare Date":
    st.header(" Vizualizare Generală a Datelor")
    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write(" Dimensiunea datasetului:", data.shape)
        if st.checkbox("Afișează primele 5 coloane și rânduri"):
            st.dataframe(data.iloc[:, :5].head())
        st.subheader(" Set complet de date:")
        st.dataframe(data)
        st.subheader(" Analiza valorilor lipsă")
        missing_vals = data.isnull().sum()
        missing_percent = (missing_vals / len(data)) * 100
        missing_df = pd.DataFrame({'Missing Values': missing_vals, 'Percentage': missing_percent})
        missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage', ascending=False)
        if not missing_df.empty:
            fig, ax = plt.subplots()
            sns.barplot(x=missing_df['Percentage'], y=missing_df.index, color="orange", ax=ax)
            ax.set_title('Procentul valorilor lipsă per coloană')
            ax.set_xlabel('Procent (%)')
            ax.set_ylabel('Coloană')
            st.pyplot(fig)
        else:
            st.success(" Nu există valori lipsă în dataset (după preprocesarea inițială).")

elif section == "Filtrare & Analiză":
    st.header(" Filtrare & Căutare")
    if "data" in st.session_state:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col_to_filter = st.selectbox("Coloană numerică pentru filtrare:", numeric_cols)
            min_val, max_val = data[col_to_filter].min(), data[col_to_filter].max()
            val_range = st.slider("Interval de valori:", float(min_val), float(max_val), (float(min_val), float(max_val)))
            filtered_data = data[(data[col_to_filter] >= val_range[0]) & (data[col_to_filter] <= val_range[1])]
            st.write(" Rezultate filtrate:")
            st.dataframe(filtered_data)
        if "Regiune" in data.columns:
            search_term = st.text_input("Introduceți o regiune pentru căutare:")
            if search_term:
                filtered = data[data["Regiune"].str.contains(search_term, case=False, na=False)]
                st.write(f" Rezultate pentru: {search_term}")
                st.dataframe(filtered)
        if "An" in data.columns:
            st.write(" Număr înregistrări pe An:")
            st.bar_chart(data["An"].value_counts().sort_index())

elif section == "Prelucrare & Grafice":
    st.header(" Prelucrare și Vizualizare Grafică")
    if "data" in st.session_state:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        default_col_analyze = "Valoare" if "Valoare" in numeric_cols else (numeric_cols[0] if numeric_cols else None)

        if default_col_analyze:
            col_to_analyze = st.selectbox("Selectați o coloană numerică pentru analiză:", numeric_cols, index=numeric_cols.index(default_col_analyze) if default_col_analyze in numeric_cols else 0)
            st.subheader(f" Statistici descriptive pentru {col_to_analyze}:")
            st.write(data[col_to_analyze].describe())
            fig1, ax1 = plt.subplots()
            ax1.hist(data[col_to_analyze].dropna(), bins=20)
            ax1.set_title(f"Distribuția valorilor pentru {col_to_analyze}")
            st.pyplot(fig1)

            if "An" in data.columns:
                if pd.api.types.is_numeric_dtype(data[col_to_analyze]):
                    grouped = data.groupby("An", as_index=False)[col_to_analyze].mean()
                    X = grouped[["An"]]
                    y = grouped[col_to_analyze]
                    model = LinearRegression()
                    future_years_list = list(range(X["An"].min(), X["An"].max() + 3))
                    future_years_df = pd.DataFrame({"An": future_years_list})
                    future_preds = model.predict(future_years_df)

                    fig2, ax2 = plt.subplots()
                    ax2.plot(X["An"], y, marker="o", label="Valori reale (medie)")
                    ax2.plot(future_years_df["An"], future_preds, linestyle="--", marker="x", label="Predicție regresie liniară")
                    ax2.set_title(f" Evoluție și Predicție Medie: {col_to_analyze} în funcție de An")
                    ax2.set_xlabel("An")
                    ax2.set_ylabel(f"Media {col_to_analyze}")
                    ax2.legend()
                    st.pyplot(fig2)
                    st.info(f" Predicție medie pentru anul {future_years_list[-1]}: *{future_preds[-1]:.2f}*")
                else:
                    st.warning(f"Coloana '{col_to_analyze}' selectată nu este numerică. Nu se poate efectua analiza de regresie temporală.")
            
            categorical_cols_for_grouping = [col for col in ['Nivel_Educatie', 'Limba_Predare', 'Regiune'] if col in data.columns]

            if categorical_cols_for_grouping and col_to_analyze:
                group_by_col = st.selectbox("Selectați o coloană categorică pentru grupare:", categorical_cols_for_grouping)
                st.subheader(f" Agregare '{col_to_analyze}' după '{group_by_col}':")
                if pd.api.types.is_numeric_dtype(data[col_to_analyze]):
                    st.dataframe(data.groupby(group_by_col).agg(
                        Suma=(col_to_analyze, 'sum'),
                        Media=(col_to_analyze, 'mean'),
                        Min=(col_to_analyze, 'min'),
                        Max=(col_to_analyze, 'max'),
                        Numar_Inregistrari=('Valoare', 'count')
                    ))
                else:
                     st.warning(f"Coloana '{col_to_analyze}' selectată nu este numerică. Nu se pot afișa agregările.")


            if "Nivel_Educatie" in data.columns and "An" in data.columns and col_to_analyze:
                 if pd.api.types.is_numeric_dtype(data[col_to_analyze]):
                    st.subheader(f" Agregare '{col_to_analyze}' după 'Nivel_Educatie' și 'An':")
                    st.dataframe(data.groupby(["Nivel_Educatie", "An"]).agg(
                        Suma=(col_to_analyze, 'sum'),
                        Media=(col_to_analyze, 'mean'),
                        Numar_Inregistrari=('Valoare', 'count')
                    ).sort_index())
                 else:
                    st.warning(f"Coloana '{col_to_analyze}' selectată nu este numerică. Nu se pot afișa agregările.")
            
        if numeric_cols:
            st.subheader(" Matricea de corelație")
            corr_data = data[numeric_cols].corr()
            fig4, ax4 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
            ax4.set_title("Matricea de corelație pentru variabilele numerice")
            st.pyplot(fig4)
        else:
            st.warning("Nu există suficiente coloane numerice pentru a calcula o matrice de corelație.")


elif section == "Codificare Date":
    st.header(" Codificare a datelor categorice")
    if "data" in st.session_state:
        data_df = st.session_state["data"]
        data_copy = data_df.copy()
        cat_cols = data_df.select_dtypes(include='object').columns.tolist()
        if not cat_cols:
            st.warning(" Nu există coloane categorice în dataset.")
        else:
            col_to_encode = st.selectbox("Selectați o coloană categorică pentru codificare:", cat_cols)
            encoding_type = st.radio("Tip de codificare:", ["Label Encoding", "One-Hot Encoding"])
            
            if encoding_type == "Label Encoding":
                encoded_col_name = col_to_encode + "_encoded"
                le = LabelEncoder()
                data_copy[encoded_col_name] = le.fit_transform(data_copy[col_to_encode].astype(str))
                st.dataframe(data_copy[[col_to_encode, encoded_col_name]].head())
                if st.button(f"Aplică Label Encoding pentru '{col_to_encode}'"):
                    st.session_state["data"][encoded_col_name] = data_copy[encoded_col_name]
                    st.success(f"Coloana '{encoded_col_name}' a fost adăugată.")

            else:
                try:
                    prefix = col_to_encode.replace(" ", "_")
                    encoded_df = pd.get_dummies(data_copy[col_to_encode], prefix=prefix, dummy_na=False)
                    new_cols = [col for col in encoded_df.columns if col not in data_copy.columns]
                    data_copy = pd.concat([data_copy, encoded_df[new_cols]], axis=1)
                    st.dataframe(data_copy[[col_to_encode] + new_cols].head())
                    if st.button(f"Aplică One-Hot Encoding pentru '{col_to_encode}'"):
                        for new_col in new_cols:
                            st.session_state["data"][new_col] = data_copy[new_col]
                        st.success(f"Coloanele One-Hot pentru '{col_to_encode}' au fost adăugate.")
                except Exception as e:
                    st.error(f"Eroare la One-Hot Encoding: {e}")


elif section == "Scalare Date":
    st.header(" Scalare a variabilelor numerice")
    if "data" in st.session_state:
        data_df = st.session_state["data"]
        numeric_cols = data_df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.warning("Nu există coloane numerice pentru scalare.")
        else:
            cols_to_scale = st.multiselect("Selectați coloane pentru scalare:", numeric_cols)
            scaler_type = st.radio("Alegeți metoda de scalare:", ["StandardScaler", "MinMaxScaler"])
            if cols_to_scale:
                data_to_scale = data_df[cols_to_scale].copy()
                scaler = StandardScaler() if scaler_type == "StandardScaler" else MinMaxScaler()
                
                try:
                    scaled_values = scaler.fit_transform(data_to_scale)
                    scaled_cols_df = pd.DataFrame(scaled_values, columns=[col + "_scaled" for col in cols_to_scale], index=data_to_scale.index)
                    
                    st.write("Date scalate (primele 5 rânduri):")
                    st.dataframe(scaled_cols_df.head())

                    if st.button("Aplică Scalarea Datelor"):
                        for col in scaled_cols_df.columns:
                            st.session_state["data"][col] = scaled_cols_df[col]
                        st.success("Scalare aplicată și coloanele adăugate în dataset.")
                        st.experimental_rerun()
                except Exception as e:
                     st.error(f"Eroare la scalarea datelor: {e}. Verificați dacă toate coloanele selectate sunt numerice și nu conțin doar NaN.")


elif section == "Outlieri":
    st.header(" Detectarea valorilor extreme (outlieri)")
    if "data" in st.session_state:
        data = st.session_state["data"]
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.warning("Nu există coloane numerice pentru analiza outlierilor.")
        else:
            default_col_outlier = "Valoare" if "Valoare" in numeric_cols else numeric_cols[0]
            col_to_check = st.selectbox("Alegeți o coloană numerică pentru analiză outlieri:", numeric_cols, index=numeric_cols.index(default_col_outlier) if default_col_outlier in numeric_cols else 0)
            
            if pd.api.types.is_numeric_dtype(data[col_to_check]):
                Q1 = data[col_to_check].quantile(0.25)
                Q3 = data[col_to_check].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[col_to_check] < lower_bound) | (data[col_to_check] > upper_bound)]
                st.write(f"Număr de outliers detectați în '{col_to_check}': {len(outliers)}")
                if not outliers.empty:
                    st.dataframe(outliers)
                else:
                    st.info("Nu au fost detectați outlieri conform metodei IQR.")
                
                fig, ax = plt.subplots()
                sns.boxplot(x=data[col_to_check], ax=ax)
                ax.set_title(f"Boxplot pentru {col_to_check}")
                st.pyplot(fig)
            else:
                st.warning(f"Coloana '{col_to_check}' selectată nu este numerică.")

elif section == "Clusterizare (Scikit-learn)":
    st.header("Clusterizare cu Scikit-learn (K-Means)")
    if "data" in st.session_state:
        data = st.session_state["data"].copy()

        st.subheader("1. Selectarea Caracteristicilor și Preprocesare")
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            st.warning("Nu există coloane numerice în setul de date pentru clusterizare.")
            st.stop()

        # Exclude 'An' by default from features unless explicitly chosen, as it might dominate clusters if not scaled.
        # Also, let's try to exclude previously encoded label columns if they are just identifiers.
        default_features = [col for col in numeric_cols if col != "An" and not col.endswith("_encoded")]
        if not default_features and numeric_cols: # Fallback if all numeric cols were excluded
            default_features = [numeric_cols[0]]
        elif not default_features:
             st.warning("Nu s-au putut selecta automat caracteristici numerice potrivite. Vă rugăm selectați manual.")
             st.stop()


        features_for_clustering = st.multiselect(
            "Selectați caracteristicile numerice pentru clusterizare:",
            options=numeric_cols,
            default=default_features
        )

        if not features_for_clustering:
            st.warning("Vă rugăm să selectați cel puțin o caracteristică pentru clusterizare.")
            st.stop()

        data_for_clustering = data[features_for_clustering].copy()
        
        # Handle missing values by dropping rows
        initial_rows = len(data_for_clustering)
        data_for_clustering.dropna(inplace=True)
        cleaned_rows = len(data_for_clustering)
        if initial_rows > cleaned_rows:
            st.info(f"{initial_rows - cleaned_rows} rânduri cu valori lipsă au fost eliminate din datele selectate pentru clusterizare.")

        if data_for_clustering.empty:
            st.error("După eliminarea valorilor lipsă, nu mai există date pentru clusterizare. Verificați caracteristicile selectate.")
            st.stop()
        
        # Scaling is important for K-Means
        # We'll use StandardScaler here, but this assumes the user hasn't already scaled them in the "Scalare Date" section.
        # A more robust approach might check if columns like `_scaled` versions already exist for selected features.
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data_for_clustering)
        scaled_features_df = pd.DataFrame(scaled_features, columns=data_for_clustering.columns, index=data_for_clustering.index)

        st.write("Datele selectate și scalate (primele 5 rânduri):")
        st.dataframe(scaled_features_df.head())

        st.subheader("2. Alegerea Numărului de Clustere (K) - Metoda Cotului (Elbow Method)")
        max_k = st.slider("Numărul maxim de clustere (K) pentru testare (Metoda Cotului):", 2, 15, 10)
        
        inertia = []
        k_range = range(1, max_k + 1)
        with st.spinner(f"Se calculează metoda cotului pentru K de la 1 la {max_k}..."):
            for k_val in k_range:
                kmeans_elbow = KMeans(n_clusters=k_val, init='k-means++', random_state=42, n_init='auto')
                kmeans_elbow.fit(scaled_features_df)
                inertia.append(kmeans_elbow.inertia_)

        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(k_range, inertia, marker='o')
        ax_elbow.set_title('Metoda Cotului pentru determinarea K optim')
        ax_elbow.set_xlabel('Număr de Clustere (K)')
        ax_elbow.set_ylabel('Inerția (WCSS - Within-Cluster Sum of Squares)')
        st.pyplot(fig_elbow)
        st.info("Căutați un 'cot' în grafic. Punctul unde adăugarea unui nou cluster nu mai reduce semnificativ inerția este adesea un K bun.")

        st.subheader("3. Aplicarea K-Means Clustering")
        chosen_k = st.number_input("Introduceți numărul de clustere (K) dorit:", min_value=2, max_value=max_k, value=3, step=1)

        if st.button("Rulează K-Means Clustering"):
            kmeans_model = KMeans(n_clusters=chosen_k, init='k-means++', random_state=42, n_init='auto')
            cluster_labels = kmeans_model.fit_predict(scaled_features_df)
            
            # Add cluster labels to the original data (or a copy of it)
            data_clustered = data.loc[scaled_features_df.index].copy() # Ensure we use indices from the cleaned/scaled data
            data_clustered['Cluster'] = cluster_labels
            st.session_state['data_clustered'] = data_clustered # Save for potential further use

            st.success(f"Clusterizare K-Means finalizată cu K={chosen_k}.")
            st.write("Date cu etichetele de cluster atribuite (primele 20 rânduri):")
            st.dataframe(data_clustered.head(20))

            # Store cluster centers (after inverse transform if needed for interpretation, but here from scaled data)
            cluster_centers_scaled = kmeans_model.cluster_centers_
            cluster_centers_df = pd.DataFrame(cluster_centers_scaled, columns=features_for_clustering)
            st.write("Centrele clusterelor (valori scalate):")
            st.dataframe(cluster_centers_df)
            
            st.subheader("4. Vizualizarea Clusterelor")
            if len(features_for_clustering) == 2:
                fig_clusters, ax_clusters = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=scaled_features_df.iloc[:, 0], y=scaled_features_df.iloc[:, 1], hue=cluster_labels, palette='viridis', ax=ax_clusters, legend='full')
                # Plot cluster centers
                ax_clusters.scatter(cluster_centers_scaled[:, 0], cluster_centers_scaled[:, 1], marker='X', s=200, color='red', label='Centre Clustere')
                ax_clusters.set_title(f'Clustere K-Means (K={chosen_k}) pentru {features_for_clustering[0]} vs {features_for_clustering[1]} (scalate)')
                ax_clusters.set_xlabel(f'{features_for_clustering[0]} (scalat)')
                ax_clusters.set_ylabel(f'{features_for_clustering[1]} (scalat)')
                ax_clusters.legend()
                st.pyplot(fig_clusters)
            elif len(features_for_clustering) > 2:
                st.info("Pentru vizualizarea clusterelor cu mai mult de 2 caracteristici, se va folosi PCA pentru reducerea dimensionalității la 2 componente.")
                pca = PCA(n_components=2, random_state=42)
                principal_components = pca.fit_transform(scaled_features_df)
                pca_df = pd.DataFrame(data=principal_components, columns=['Componenta Principală 1', 'Componenta Principală 2'], index=scaled_features_df.index)
                pca_df['Cluster'] = cluster_labels

                fig_pca_clusters, ax_pca_clusters = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x='Componenta Principală 1', y='Componenta Principală 2', hue='Cluster', data=pca_df, palette='viridis', ax=ax_pca_clusters, legend='full')
                # It's harder to meaningfully plot centers of PCA-transformed data directly from original scaled centers
                # For simplicity, we'll skip plotting PCA centers here or calculate them on PCA data.
                ax_pca_clusters.set_title(f'Clustere K-Means (K={chosen_k}) vizualizate cu PCA')
                st.pyplot(fig_pca_clusters)
                
                explained_variance_ratio = pca.explained_variance_ratio_
                st.write(f"Varianța explicată de Componenta Principală 1: {explained_variance_ratio[0]:.2%}")
                st.write(f"Varianța explicată de Componenta Principală 2: {explained_variance_ratio[1]:.2%}")
                st.write(f"Varianța totală explicată de cele 2 componente: {sum(explained_variance_ratio):.2%}")

            else: # Only 1 feature
                st.info("Vizualizarea clusterelor pentru o singură caracteristică este limitată. Se afișează o histogramă colorată pe cluster.")
                fig_hist_clusters, ax_hist_clusters = plt.subplots(figsize=(10, 6))
                # Create a temporary df for plotting histogram with hue
                temp_df_hist = pd.DataFrame({
                    features_for_clustering[0]: scaled_features_df.iloc[:,0],
                    'Cluster': cluster_labels
                })
                sns.histplot(data=temp_df_hist, x=features_for_clustering[0], hue='Cluster', palette='viridis', multiple="stack", ax=ax_hist_clusters)
                ax_hist_clusters.set_title(f'Distribuția caracteristicii {features_for_clustering[0]} (scalat) pe clustere (K={chosen_k})')
                st.pyplot(fig_hist_clusters)

            # Display summary statistics per cluster
            st.subheader("5. Statistici Descriptive pe Cluster")
            # Use original data values (before scaling) for interpretability of stats, but add Cluster column
            original_data_with_clusters = data.loc[scaled_features_df.index].copy()
            original_data_with_clusters['Cluster'] = cluster_labels
            
            st.write(f"Medii ale caracteristicilor (originale, nescalate) pentru fiecare cluster (K={chosen_k}):")
            cluster_summary = original_data_with_clusters.groupby('Cluster')[features_for_clustering].mean()
            st.dataframe(cluster_summary)
            
            st.write(f"Numărul de observații în fiecare cluster:")
            st.dataframe(original_data_with_clusters['Cluster'].value_counts().sort_index().rename("Număr Observații"))

    else:
        st.error("Setul de date nu este încărcat. Mergeți la secțiunea corespunzătoare pentru a încărca datele.")

elif section == "Regresie Logistică (Scikit-learn)":
    st.header("Regresie Logistică cu Scikit-learn")
    if "data" in st.session_state:
        data_full = st.session_state["data"].copy() # Work with a copy of the full data

        st.subheader("1. Definirea Variabilei Țintă Binare")
        target_creation_method = st.radio(
            "Cum doriți să definiți variabila țintă?",
            ("Binarizează o coloană numerică", "Utilizează o coloană categorică existentă (selectează 2 categorii)"),
            key="logreg_target_method"
        )

        target_col_name = "target_binary"
        # Initialize processed_data here to ensure it's always available
        processed_data = data_full.copy()
        # Initialize y before the conditional blocks
        y = None

        if target_creation_method == "Binarizează o coloană numerică":
            numeric_cols_for_target = data_full.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols_for_target:
                st.warning("Nu există coloane numerice pentru a binariza.")
                st.stop()
            
            # Ensure default index is valid
            default_idx_binarize = 0
            if 'Valoare' in numeric_cols_for_target:
                default_idx_binarize = numeric_cols_for_target.index('Valoare')
            elif numeric_cols_for_target:
                 pass # default_idx_binarize is already 0
            else: # Should not happen due to earlier check, but as a safeguard
                st.error("Eroare neașteptată: Nicio coloană numerică pentru binarizare, deși verificarea a trecut.")
                st.stop()

            col_to_binarize = st.selectbox("Selectați coloana numerică pentru binarizare:", numeric_cols_for_target, index=default_idx_binarize, key="logreg_col_binarize")
            
            if col_to_binarize and not data_full[col_to_binarize].empty:
                default_threshold = data_full[col_to_binarize].median()
                threshold = st.number_input(f"Introduceți pragul pentru '{col_to_binarize}' (valorile > pragului vor fi 1, restul 0):", value=float(default_threshold), key="logreg_threshold")
                
                processed_data.dropna(subset=[col_to_binarize], inplace=True)
                if not processed_data.empty:
                    processed_data[target_col_name] = (processed_data[col_to_binarize] > threshold).astype(int)
                    y = processed_data[target_col_name]
                    st.write(f"Variabila țintă '{target_col_name}' creată. Distribuția:")
                    st.write(y.value_counts())
                else:
                    st.warning("Nu mai există date după eliminarea NaNs din coloana selectată pentru binarizare.")
                    st.stop()
            else:
                st.warning("Coloana selectată pentru binarizare este goală sau invalidă.")
                st.stop()

        elif target_creation_method == "Utilizează o coloană categorică existentă (selectează 2 categorii)":
            all_cols = data_full.columns.tolist()
            potential_cat_cols = [col for col in all_cols if data_full[col].dtype == 'object' or (data_full[col].dtype == 'int64' and data_full[col].nunique() < 20 and col not in ['An', 'Valoare'])]
            
            if not potential_cat_cols:
                st.warning("Nu există coloane categorice potrivite pentru a defini variabila țintă.")
                st.stop()
            
            col_for_binary_target = st.selectbox("Selectați coloana categorică:", list(set(potential_cat_cols)), key="logreg_col_cat_target")
            
            processed_data.dropna(subset=[col_for_binary_target], inplace=True)
            if processed_data.empty:
                st.warning(f"Nu mai există date după eliminarea NaNs din coloana categorică '{col_for_binary_target}'.")
                st.stop()
            
            unique_categories = processed_data[col_for_binary_target].unique().tolist()
            
            if len(unique_categories) < 2:
                st.warning(f"Coloana '{col_for_binary_target}' are mai puțin de 2 categorii unice după eliminarea NaNs. Alegeți altă coloană.")
                st.stop()
            
            positive_class_category = st.selectbox(f"Selectați categoria pentru clasa pozitivă (1) din '{col_for_binary_target}':", unique_categories, key="logreg_pos_cat")
            negative_class_categories = [cat for cat in unique_categories if cat != positive_class_category]
            
            if not negative_class_categories:
                st.warning("Este necesară cel puțin o altă categorie pentru clasa negativă.")
                st.stop()
                
            negative_class_category = st.selectbox(f"Selectați categoria pentru clasa negativă (0) din '{col_for_binary_target}':", negative_class_categories, key="logreg_neg_cat")

            processed_data = processed_data[processed_data[col_for_binary_target].isin([positive_class_category, negative_class_category])].copy()
            if not processed_data.empty:
                processed_data[target_col_name] = (processed_data[col_for_binary_target] == positive_class_category).astype(int)
                y = processed_data[target_col_name]
                st.write(f"Variabila țintă '{target_col_name}' creată. Distribuția:")
                st.write(y.value_counts())
            else:
                st.warning(f"Nu mai există date după filtrarea categoriilor selectate pentru '{col_for_binary_target}'.")
                st.stop()

        # Common checks for y after its definition
        if y is not None:
            if y.empty:
                st.error("Variabila țintă este goală. Verificați pașii de definire.")
                st.stop()
            target_counts = y.value_counts(normalize=True) * 100
            if target_counts.min() < 5 or target_counts.max() > 95: # Adjusted imbalance threshold
                st.warning(f"Atenție: Variabila țintă este dezechilibrată: Clasa 0: {target_counts.get(0,0):.2f}%, Clasa 1: {target_counts.get(1,0):.2f}%. Acest lucru poate afecta performanța modelului.")
            if len(y.unique()) < 2:
                st.error(f"Variabila țintă '{target_col_name}' rezultată are o singură clasă. Regresia logistică necesită două clase. Verificați setările.")
                st.stop()
        else:
            st.info("Definiți variabila țintă pentru a continua.")
            st.stop()

        st.subheader("2. Selectarea Caracteristicilor (Variabile Independente)")
        # Determine original column used for target creation to exclude it from features
        col_used_for_target_creation = None
        if target_creation_method == "Binarizează o coloană numerică":
            col_used_for_target_creation = col_to_binarize
        elif target_creation_method == "Utilizează o coloană categorică existentă (selectează 2 categorii)":
            col_used_for_target_creation = col_for_binary_target
            
        available_features = [col for col in processed_data.columns if col != target_col_name and col != col_used_for_target_creation]
        
        # Default features: all numeric excluding 'An' and those ending in '_scaled' or '_encoded' (if not the target source itself)
        default_logreg_features = []
        for f in available_features:
            if pd.api.types.is_numeric_dtype(processed_data[f]) and f != 'An' and not f.endswith('_scaled') and not f.endswith('_encoded'):
                default_logreg_features.append(f)
            elif pd.api.types.is_object_dtype(processed_data[f]): # Also include object types as they can be OHE
                default_logreg_features.append(f)
        
        if not default_logreg_features and available_features:
            default_logreg_features = [available_features[0]]

        selected_features = st.multiselect("Selectați caracteristicile pentru model:", options=available_features, default=default_logreg_features, key="logreg_features")

        if not selected_features:
            st.warning("Vă rugăm să selectați cel puțin o caracteristică.")
            st.stop()

        X = processed_data[selected_features].copy()
        # y was already defined and aligned with processed_data from which X is derived.
        # However, if X had further NaNs dropped, y needs re-alignment.
        # This will be handled in the next chunk (preprocessing of X).

        st.write("Variabila țintă (y) și Caracteristicile selectate (X) înainte de preprocesare detaliată (primele 5 rânduri):")
        st.write("Y (Țintă):")
        st.dataframe(y.head())
        st.write("X (Caracteristici):")
        st.dataframe(X.head())
        
        # Placeholder for next parts (preprocessing X, train-test split, model, eval)
        # st.info("Următorii pași: Preprocesarea caracteristicilor X (valori lipsă, OHE, scalare), împărțirea datelor, antrenarea și evaluarea modelului.")

        st.subheader("3. Preprocesarea Caracteristicilor și Împărțirea Datelor")
        # Preprocessing for selected features in X
        X_processed = X.copy()
        
        # Handle missing values
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    median_val = X_processed[col].median()
                    X_processed[col].fillna(median_val, inplace=True)
                    st.write(f"Info: Valorile lipsă din coloana numerică '{col}' au fost înlocuite cu mediana ({median_val:.2f}).")
                else: # Categorical - should ideally be handled before OHE, or imputed with mode
                    mode_val = X_processed[col].mode()
                    if not mode_val.empty:
                        X_processed[col].fillna(mode_val[0], inplace=True)
                        st.write(f"Info: Valorile lipsă din coloana categorică '{col}' au fost înlocuite cu modul ('{mode_val[0]}').")
                    else:
                        # If mode is empty (e.g. all NaN column), drop it or handle as error
                        X_processed.drop(columns=[col], inplace=True)
                        st.warning(f"Atenție: Coloana '{col}' avea doar valori lipsă și a fost eliminată.")
        
        # Re-align y with X_processed in case rows were dropped (though current imputation avoids row drops)
        # This step is more critical if opting to drop rows with NaNs.
        # y = y.loc[X_processed.index] # Ensure alignment if any rows were dropped from X_processed
        # Check if X_processed or y became empty after NaN handling
        if X_processed.empty or y.loc[X_processed.index].empty: # Check y alignment here
            st.error("Setul de date a devenit gol după gestionarea valorilor lipsă. Verificați datele inițiale și selecția de caracteristici.")
            st.stop()
        y_aligned = y.loc[X_processed.index] # Align y with X_processed *after* NaN handling in X_processed

        # One-Hot Encode categorical features in X_processed
        categorical_features_in_X = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_features_in_X:
            st.write(f"Info: Se aplică One-Hot Encoding pentru coloanele categorice: {categorical_features_in_X}")
            X_processed = pd.get_dummies(X_processed, columns=categorical_features_in_X, drop_first=True)
            st.write("Caracteristici după One-Hot Encoding (primele 5 rânduri):")
            st.dataframe(X_processed.head())

        # Scale numerical features in X_processed (ensure they are all numeric after OHE)
        numerical_features_in_X = X_processed.select_dtypes(include=np.number).columns.tolist()
        if numerical_features_in_X:
            scaler_logreg = StandardScaler()
            X_processed[numerical_features_in_X] = scaler_logreg.fit_transform(X_processed[numerical_features_in_X])
            st.write("Caracteristici numerice scalate (primele 5 rânduri):")
            st.dataframe(X_processed[numerical_features_in_X].head())
        else:
            st.warning("Nu există caracteristici numerice de scalat după One-Hot Encoding.")

        if X_processed.empty:
            st.error("Nu mai există caracteristici după preprocesare. Verificați selecția inițială.")
            st.stop()
            
        st.subheader("4. Antrenarea Modelului și Evaluarea")
        test_size = st.slider("Procentul setului de testare:", 0.1, 0.5, 0.25, 0.05, key="logreg_test_size")
        random_state_logreg = st.number_input("Seed pentru reproductibilitate (Random State):", value=42, step=1, key="logreg_random_state")

        if st.button("Antrenează și Evaluează Modelul de Regresie Logistică", key="logreg_train_eval"):
            if X_processed.empty or len(X_processed) < 2 or y_aligned.empty or len(y_aligned.unique()) < 2:
                 st.error("Date insuficiente sau problemă cu variabila țintă/caracteristici după preprocesare. Verificați pașii anteriori.")
                 st.stop()

            X_train, X_test, y_train, y_test = train_test_split(X_processed, y_aligned, test_size=test_size, random_state=random_state_logreg, stratify=y_aligned if len(y_aligned.unique()) > 1 else None)

            if len(y_train.unique()) < 2 or (len(y_test.unique()) < 2 and len(y_test) > 0) : # Check y_test only if it's not empty
                st.warning("Setul de antrenament sau testare conține o singură clasă după divizare. Acest lucru poate fi din cauza unui set de date mic, a unei distribuții dezechilibrate extreme sau a unei divizări nefericite. Încercați să ajustați dimensiunea setului de testare, verificați definirea variabilei țintă sau datele inițiale.")
                # Optionally stop if y_test is problematic for evaluation, or proceed with caution
                if len(y_test.unique()) < 2 and len(y_test) > 0:
                    st.error("Setul de testare nu conține ambele clase, evaluarea completă nu este posibilă.")
                    st.stop()
            
            log_reg_model = LogisticRegression(random_state=random_state_logreg, solver='liblinear', class_weight='balanced')
            log_reg_model.fit(X_train, y_train)

            y_pred_train = log_reg_model.predict(X_train)
            y_pred_test = log_reg_model.predict(X_test)
            
            st.write(f"**Acuratețe pe setul de antrenament:** {accuracy_score(y_train, y_pred_train):.4f}")
            if len(y_test) > 0 and len(set(y_test)) > 1 : # Ensure y_test is usable for metrics
                y_pred_proba_test = log_reg_model.predict_proba(X_test)[:, 1]
                st.write(f"**Acuratețe pe setul de testare:** {accuracy_score(y_test, y_pred_test):.4f}")

                st.subheader("Matricea de Confuzie (Set de Testare)")
                cm = confusion_matrix(y_test, y_pred_test)
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=log_reg_model.classes_, yticklabels=log_reg_model.classes_)
                ax_cm.set_xlabel('Prezis')
                ax_cm.set_ylabel('Actual')
                ax_cm.set_title('Matrice de Confuzie')
                st.pyplot(fig_cm)

                st.subheader("Raport de Clasificare (Set de Testare)")
                try:
                    report = classification_report(y_test, y_pred_test, output_dict=False, zero_division=0)
                    st.text(report)
                except ValueError as e_report:
                    st.warning(f"Nu s-a putut genera raportul de clasificare: {e_report}")

                st.subheader("Curba ROC (Set de Testare)")
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_test)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curba ROC (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Rata Fals Pozitivelor (FPR)')
                ax_roc.set_ylabel('Rata Adevărat Pozitivelor (TPR)')
                ax_roc.set_title('Caracteristica de Operare a Receptorului (ROC)')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.warning("Setul de testare este gol sau conține o singură clasă. Evaluarea detaliată pe setul de testare nu este posibilă.")

            st.subheader("Coeficienții Modelului")
            if hasattr(log_reg_model, 'coef_') and X_processed.shape[1] > 0:
                try:
                    coefficients = pd.DataFrame({
                        'Caracteristică': X_processed.columns, # Use columns from X_processed
                        'Coeficient': log_reg_model.coef_[0]
                    }).sort_values(by='Coeficient', ascending=False)
                    st.dataframe(coefficients)
                    st.info("Coeficienții indică modificarea log-odds a variabilei țintă pentru o unitate de modificare în caracteristică, menținând celelalte caracteristici constante.")
                except Exception as e_coef:
                    st.warning(f"Nu s-au putut afișa coeficienții: {e_coef}")
            else:
                st.warning("Modelul nu are coeficienți disponibili sau nu există caracteristici procesate.")

    else:
        st.error("Setul de date nu este încărcat. Mergeți la secțiunea corespunzătoare pentru a încărca datele.")

elif section == "Regresie Multiplă (Statsmodels)":
    st.header("Regresie Multiplă cu Statsmodels (OLS)")
    if "data" in st.session_state:
        data_sm = st.session_state["data"].copy()

        st.subheader("1. Selectarea Variabilelor")
        numeric_cols_sm = data_sm.select_dtypes(include=np.number).columns.tolist()
        all_cols_sm = data_sm.columns.tolist()

        if not numeric_cols_sm:
            st.warning("Nu există coloane numerice pentru a selecta variabila dependentă.")
            st.stop()

        # Ensure 'Valoare' is default if available, otherwise first numeric
        default_dependent_var_idx = 0
        if 'Valoare' in numeric_cols_sm:
            default_dependent_var_idx = numeric_cols_sm.index('Valoare')
        
        dependent_var = st.selectbox(
            "Selectați Variabila Dependentă (Y - numerică):", 
            numeric_cols_sm, 
            index=default_dependent_var_idx,
            key="sm_dependent_var"
        )

        # Exclude dependent variable and 'An' (often not a direct predictor in this context unless time series)
        potential_independent_vars = [col for col in all_cols_sm if col != dependent_var and col != 'An']
        
        # Try to pre-select some sensible defaults for independent variables
        default_independent_vars = []
        for col in potential_independent_vars:
            if pd.api.types.is_numeric_dtype(data_sm[col]) and not col.endswith('_scaled') and not col.endswith('_encoded'):
                default_independent_vars.append(col)
            elif data_sm[col].dtype == 'object' and data_sm[col].nunique() < 10: # Select categoricals with few unique values
                 default_independent_vars.append(col)
        # Ensure at least one default if possible
        if not default_independent_vars and potential_independent_vars:
            default_independent_vars = [potential_independent_vars[0]]
            

        independent_vars = st.multiselect(
            "Selectați Variabilele Independente (X):", 
            potential_independent_vars,
            default=default_independent_vars,
            key="sm_independent_vars"
        )

        if not dependent_var or not independent_vars:
            st.warning("Vă rugăm să selectați atât variabila dependentă, cât și cel puțin o variabilă independentă.")
            st.stop()

        st.subheader("2. Preprocesarea Datelor")
        # Prepare data for statsmodels
        data_reg = data_sm[[dependent_var] + independent_vars].copy()
        
        # Handle missing values by dropping rows for simplicity in OLS
        initial_rows_reg = len(data_reg)
        data_reg.dropna(inplace=True)
        cleaned_rows_reg = len(data_reg)
        if initial_rows_reg > cleaned_rows_reg:
            st.info(f"{initial_rows_reg - cleaned_rows_reg} rânduri cu valori lipsă au fost eliminate din datele selectate pentru regresie.")

        if data_reg.empty or len(data_reg) < len(independent_vars) + 2: # Need enough data points
            st.error("Date insuficiente pentru regresie după eliminarea valorilor lipsă sau prea puține observații. Verificați selecția de variabile.")
            st.stop()

        # One-Hot Encode categorical independent variables
        X_reg = data_reg[independent_vars]
        y_reg = data_reg[dependent_var]
        
        categorical_cols_reg = X_reg.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols_reg:
            st.write(f"Info: Se aplică One-Hot Encoding pentru variabilele independente categorice: {categorical_cols_reg}")
            X_reg = pd.get_dummies(X_reg, columns=categorical_cols_reg, drop_first=True)
            st.write("Variabile independente după One-Hot Encoding (primele 5 rânduri):")
            st.dataframe(X_reg.head())
        
        # Add constant for the intercept term
        X_reg_with_const = sm.add_constant(X_reg)
        
        if X_reg_with_const.empty:
            st.error("Setul de variabile independente (X) este gol după preprocesare.")
            st.stop()
        if y_reg.empty:
            st.error("Variabila dependentă (Y) este goală după preprocesare.")
            st.stop()

        st.write("Variabila Dependentă (Y) (primele 5 rânduri):")
        st.dataframe(y_reg.head())
        st.write("Variabile Independente (X) cu constantă (primele 5 rânduri):")
        st.dataframe(X_reg_with_const.head())

        st.subheader("3. Rularea Modelului OLS și Rezultate")
        if st.button("Rulează Regresia Multiplă (OLS)", key="sm_run_ols"):
            try:
                ols_model = sm.OLS(y_reg, X_reg_with_const.astype(float)) # Ensure X is float for OLS
                ols_results = ols_model.fit()
                
                st.write("Rezumatul Modelului de Regresie OLS:")
                # The summary_as_text() can be quite wide. Using st.text might be better for scroll.
                st.text(ols_results.summary().as_text())
                
                st.info("**Interpretarea Rezultatelor Principale:**\n"
                        "- **R-squared (R-pătrat):** Procentul de varianță în variabila dependentă explicat de model.\n"
                        "- **Adj. R-squared:** R-pătrat ajustat pentru numărul de predictori.\n"
                        "- **coef:** Coeficienții pentru fiecare variabilă. Indică schimbarea medie în Y pentru o unitate de schimbare în X, menținând ceilalți predictori constanți.\n"
                        "- **P>|t| (p-value):** Probabilitatea de a observa un coeficient la fel de mare (sau mai mare) dacă ipoteza nulă (coeficientul este zero) ar fi adevărată. Valori mici (ex: < 0.05) sugerează că predictorul este semnificativ statistic.\n"
                        "- **F-statistic & Prob (F-statistic):** Testează semnificația generală a modelului.")

                # Optional: Display residuals plot or other diagnostics if desired
                # fig_resid, ax_resid = plt.subplots()
                # sns.residplot(x=ols_results.fittedvalues, y=ols_results.resid, lowess=True, ax=ax_resid, line_kws={'color': 'red', 'lw': 1})
                # ax_resid.set_title('Plot Reziduale')
                # ax_resid.set_xlabel('Valori Ajustate')
                # ax_resid.set_ylabel('Reziduale')
                # st.pyplot(fig_resid)

            except Exception as e_ols:
                st.error(f"Eroare la rularea modelului OLS: {e_ols}")
                st.error("Cauze posibile: multicoliniaritate perfectă (predictori puternic corelați), date insuficiente după eliminarea NaN, tipuri de date incorecte neconvertite la float.")

    else:
        st.error("Setul de date nu este încărcat. Mergeți la secțiunea corespunzătoare pentru a încărca datele.")

# SAS Runner Section
elif section == "SAS Runner" or section == "SAS Runner (saspy indisponibil)":
    st.header("SAS Runner")

    if not SAS_AVAILABLE:
        st.error("Biblioteca `saspy` nu este instalată sau configurată corect. Funcționalitatea SAS nu este disponibilă.")
        st.markdown("Vă rugăm să vă asigurați că aveți SAS instalat și `saspy` configurat corespunzător. Pași:")
        st.markdown("1. `pip install saspy`")
        st.markdown("2. Configurați `saspy` (de ex., creați un fișier `sascfg_personal.py`). Consultați [documentația saspy](https://sassoftware.github.io/saspy/install.html#configuration).")
        st.markdown("3. Asigurați-vă că SAS este accesibil (de ex., serverul SAS este pornit dacă folosiți o conexiune la distanță).")
        st.stop()
    
    st.info("**Notă Importantă:** Pentru a utiliza această secțiune, trebuie să:\n" 
            "1. Extrageți manual codul SAS pur din fișierele .docx și salvați-l ca fișiere `.sas`.\n" 
            "2. Asigurați-vă că `saspy` este corect configurat pentru a se conecta la instalarea dvs. SAS.")

    uploaded_sas_file = st.file_uploader("Încărcați un fișier .sas", type=["sas"])
    sas_code_description = st.text_area("Descriere scurtă a scriptului SAS (opțional)")

    if uploaded_sas_file is not None:
        try:
            sas_code_content = uploaded_sas_file.read().decode('utf-8')
            st.subheader("Conținut Script SAS:")
            st.code(sas_code_content, language='sas')

            if st.button("Rulează Scriptul SAS"):
                with st.spinner("Se conectează la SAS și se execută scriptul..."):
                    sas_session = None # Initialize to ensure it's in scope for finally
                    try:
                        # Attempt to start a SAS session using default configuration
                        st.info("Încercare de inițializare sesiune SAS... Verificați configurația saspy dacă acest pas eșuează.")
                        sas_session = saspy.SASsession() # You might need to specify a config name: saspy.SASsession(cfgname='your_config_name')
                        st.success("Conectat la SAS cu succes!")

                        # Submit the SAS code
                        # For complex scripts with multiple steps, log and lst will be interleaved.
                        # ODS HTML5 output can be richer if generated.
                        result = sas_session.submit(sas_code_content)
                        
                        st.subheader("Log SAS:")
                        # saslog() returns a string which can be very long
                        sas_log = result.get('LOG', 'Logul SAS nu a putut fi recuperat.') 
                        if isinstance(sas_log, bytes):
                           sas_log = sas_log.decode('utf-8', errors='replace')
                        st.text_area("SAS Log", value=sas_log, height=300)

                        st.subheader("Output Listing (LST):")
                        # saslst() returns a string
                        sas_lst = result.get('LST', 'Output-ul SAS (LST) nu a putut fi recuperat sau nu a fost generat.')
                        if isinstance(sas_lst, bytes):
                           sas_lst = sas_lst.decode('utf-8', errors='replace')
                        if sas_lst.strip():
                           st.text_area("SAS Listing", value=sas_lst, height=400)
                        else:
                           st.info("Nu s-a generat output de tip Listing (LST) sau este gol.")
                        
                        # Check for ODS HTML5 output (often named 'results')
                        # saspy can return this as HTML that can be rendered by Streamlit
                        html_output = sas_session.HTML5_output
                        if html_output:
                            st.subheader("Output HTML5 (ODS):")
                            st.components.v1.html(html_output, height=600, scrolling=True)
                            sas_session.HTML5_output_close() # Clear for next run
                        else:
                            st.info("Nu s-a generat output HTML5 (ODS) sau nu a fost capturat.")

                    except saspy.sasexceptions.SASIOConnectionError as e_conn:
                        st.error(f"Eroare de Conexiune SAS I/O: {e_conn}")
                        st.error("Verificați dacă serverul SAS este pornit și accesibil și dacă configurația saspy (ex: `sascfg_personal.py`) este corectă.")
                        st.markdown("**Cauze comune:** Problema cu calea către executabilul SAS în `sascfg_personal.py`, SAS licențiat nefuncțional, serviciu SAS (remote/VM) nepornit.")
                    except saspy.sasexceptions.SASConfigNotFoundError as e_cfg:
                        st.error(f"Eroare Critică de Configurare saspy: {e_cfg}")
                        st.error("Nu s-a găsit fișierul de configurare saspy (`sascfg.py` sau `sascfg_personal.py`) sau configurația specificată.")
                        st.markdown("Vă rugăm creați sau verificați fișierul `sascfg_personal.py` în directorul home sau într-un loc accesibil Python.")
                    except Exception as e_startup: # Catching a more general exception for SAS session startup problems
                        st.error(f"Eroare la Pornirea sau în Timpul Sesiunii SAS: {e_startup}")
                        st.error("Acest lucru indică o problemă cu instalarea SAS, configurația saspy, sau codul SAS executat.")
                        st.error("Verificați consola pentru mesaje de diagnosticare detaliate de la saspy.")
                    except Exception as e:
                        st.error(f"A apărut o eroare generală în secțiunea SAS: {e}")
                    finally:
                        if sas_session:
                            st.write("Se închide sesiunea SAS...")
                            sas_session.endsas()
                            st.success("Sesiunea SAS a fost închisă.")
        except Exception as e_upload:
            st.error(f"Eroare la citirea fișierului SAS încărcat: {e_upload}")

    else:
        st.info("Vă rugăm să încărcați un fișier .sas și apoi să apăsați butonul de rulare.")

elif section == "Demonstrare Facilități SAS":
    st.header("Demonstrare Facilități SAS")
    st.markdown("""Această pagină demonstrează diverse funcționalități SAS, rulând exemple de cod SAS
    direct din Streamlit folosind `saspy`. Asigurați-vă că `saspy` este configurat corect și conectat
    la o instanță SAS funcțională.""")

    if not SAS_AVAILABLE:
        st.error("Biblioteca `saspy` nu este instalată sau configurată corect. Funcționalitatea SAS nu este disponibilă.")
        st.markdown("Vă rugăm să urmați instrucțiunile din pagina `SAS Runner` pentru configurare.")
        st.stop()

    sas_session_demo = None
    try:
        # Initialize SAS session for this page
        if 'sas_session_demo' not in st.session_state or st.session_state.sas_session_demo is None:
            with st.spinner("Se inițializează sesiunea SAS pentru demonstrații..."):
                st.info("Încercare de inițializare sesiune SAS demo... Verificați configurația saspy dacă acest pas eșuează.")
                st.session_state.sas_session_demo = saspy.SASsession()
            st.success("Sesiunea SAS pentru demonstrații a fost inițializată.")
        
        sas_session_demo = st.session_state.sas_session_demo

        # Helper function to run SAS code and display results
        def run_sas_code(code_to_run, title, description=""):
            st.subheader(title)
            if description:
                st.markdown(description)
            
            st.markdown("##### Cod SAS:")
            st.code(code_to_run, language='sas')

            if st.button(f"Rulează Exemplul: {title}"):
                with st.spinner(f"Se execută: {title}..."):
                    try:
                        result = sas_session_demo.submit(code_to_run)
                        
                        log = result.get('LOG', 'Logul SAS nu a putut fi recuperat.')
                        if isinstance(log, bytes): log = log.decode('utf-8', errors='replace')
                        
                        lst = result.get('LST', 'Output-ul SAS (LST) nu a putut fi recuperat sau nu a fost generat.')
                        if isinstance(lst, bytes): lst = lst.decode('utf-8', errors='replace')

                        st.markdown("##### Log SAS:")
                        st.text_area(f"Log_{title.replace(' ', '_')}", value=log, height=200, key=f"log_{title}")

                        if lst.strip():
                            st.markdown("##### Listing (LST) SAS:")
                            st.text_area(f"LST_{title.replace(' ', '_')}", value=lst, height=300, key=f"lst_{title}")
                        else:
                            st.info("Nu s-a generat output de tip Listing (LST) sau este gol.")
                        
                        html_output = sas_session_demo.HTML5_output
                        if html_output:
                            st.markdown("##### Output HTML5 (ODS):")
                            st.components.v1.html(html_output, height=500, scrolling=True)
                            sas_session_demo.HTML5_output_close()


                    except Exception as e_run:
                        st.error(f"Eroare la rularea exemplului '{title}': {e_run}")
                        st.error("Verificați log-ul SAS de mai sus pentru detalii specifice erorii din codul SAS.")
            st.markdown("---")


        st.sidebar.info("Navigați prin exemplele SAS de mai jos.")

        # 1. Crearea unui set de date SAS din fisiere externe (inline data)
        sas_code_inline_data = """
/* 1. Crearea unui set de date SAS folosind date inline (datalines/cards) */
DATA studenti_info;
    INPUT Nume $ Varsta Grupa $;
    CARDS;
Popescu 21 A1
Ionescu 22 A2
Georgescu 20 A1
Vasilescu 23 A3
;
RUN;

PROC PRINT DATA=studenti_info;
    TITLE '1. Set de date SAS creat cu date inline';
RUN;
TITLE; /* Clear title */
"""
        run_sas_code(sas_code_inline_data, "Creare Set Date (Datalines)")

        # 2. Crearea si folosirea de formate definite de utilizator
        sas_code_formats = """
/* 2. Crearea si folosirea de formate definite de utilizator */
PROC FORMAT;
    VALUE $grupafmt
        'A1' = 'Grupa Excelenta'
        'A2' = 'Grupa Buna'
        'A3' = 'Grupa Medie'
        OTHER = 'Alta Grupa';
    VALUE varstafmt
        LOW - 20 = 'Tanar'
        21 - 22  = 'Mediu'
        23 - HIGH = 'Experimentat';
RUN;

DATA studenti_formatati;
    SET studenti_info; /* Assuming studenti_info exists from previous step */
    FORMAT Grupa $grupafmt. Varsta varstafmt.;
RUN;

PROC PRINT DATA=studenti_formatati;
    TITLE '2. Set de date cu formate definite de utilizator';
RUN;
TITLE;
"""
        run_sas_code(sas_code_formats, "Formate Definite de Utilizator")

        # 3. Procesarea iterativă si conditională a datelor
        sas_code_iterative_conditional = """
/* 3. Procesarea iterativă si conditională a datelor */
DATA bonus_studenti;
    SET studenti_info; /* Assuming studenti_info exists */
    Bonus = 0;
    IF Varsta > 21 THEN Bonus = 100;
    ELSE Bonus = 50;

    /* Exemplu iterativ: calcul suma varstelor (just for demo) */
    /* This is not a typical use case for DO loop in data step for this scenario, but demonstrates syntax */
    TotalVarsta = 0;
    DO i = 1 TO Varsta;
        TotalVarsta = TotalVarsta + 1; /* Simple increment, actual age is already in Varsta */
    END;
    DROP i; /* Drop loop counter */
    /* More practical iterative example would be array processing */
RUN;

PROC PRINT DATA=bonus_studenti;
    TITLE '3. Procesare iterativă și condițională';
RUN;
TITLE;
"""
        run_sas_code(sas_code_iterative_conditional, "Procesare Iterativă și Condițională")

        # 4. Crearea de subseturi de date
        sas_code_subsets = """
/* 4. Crearea de subseturi de date */
/* Metoda 1: Folosind IF in DATA step */
DATA studenti_tineri;
    SET studenti_info; /* Assuming studenti_info exists */
    IF Varsta <= 21;
    TITLE '4.1 Subset: Studenti cu varsta <= 21 (IF in DATA step)';
RUN;
PROC PRINT DATA=studenti_tineri; RUN;
TITLE;

/* Metoda 2: Folosind clauza WHERE in PROC PRINT (sau alte PROC-uri) */
TITLE '4.2 Subset: Studenti din grupa A1 (WHERE in PROC PRINT)';
PROC PRINT DATA=studenti_info;
    WHERE Grupa = 'A1';
RUN;
TITLE;
"""
        run_sas_code(sas_code_subsets, "Creare Subseturi de Date")

        # 5. Utilizarea de functii SAS
        sas_code_functions = """
/* 5. Utilizarea de functii SAS */
DATA date_modificate;
    LungimeNume = LENGTH('Popescu Ion'); /* Functie caracter */
    DataCurenta = TODAY(); /* Functie data */
    Format DataCurenta DATE9.;
    ValoareAbs = ABS(-150); /* Functie numerica */
    MedieNote = MEAN(9, 7, 10); /* Functie statistica */
    SubNume = SUBSTR('Popescu Ion', 1, 7); /* Extrage substring */
    NumeMajuscule = UPCASE('ionescu'); /* Conversie la majuscule */
RUN;

PROC PRINT DATA=date_modificate;
    TITLE '5. Exemple de functii SAS';
RUN;
TITLE;
"""
        run_sas_code(sas_code_functions, "Utilizare Funcții SAS")
        
        # 6. Combinarea seturilor de date (SAS & SQL)
        sas_code_combine = """
/* 6. Combinarea seturilor de date */
/* Seturi de date initiale */
DATA angajati_dept1;
    INPUT ID_Angajat $ Nume $ Departament $;
    CARDS;
E01 Alex Vanzari
E02 Maria Marketing
;
RUN;

DATA angajati_dept2;
    INPUT ID_Angajat $ Nume $ Departament $ Salariu;
    CARDS;
E03 Ion Vanzari 1500
E04 Ana IT 2500
E02 Maria Marketing 2200 /* Duplicate ID for merge example */
;
RUN;

/* Concatenare (SET statement) */
DATA toti_angajatii_set;
    SET angajati_dept1 angajati_dept2;
    TITLE '6.1 Concatenare cu SET';
RUN;
PROC PRINT DATA=toti_angajatii_set; RUN;
TITLE;

/* Interclasare (MERGE statement) - necesita sortare */
PROC SORT DATA=angajati_dept1; BY ID_Angajat; RUN;
PROC SORT DATA=angajati_dept2; BY ID_Angajat; RUN;
DATA angajati_merged;
    MERGE angajati_dept1(IN=a) angajati_dept2(IN=b RENAME=(Departament=DeptSalariu));
    BY ID_Angajat;
    IF a; /* Pastreaza doar cei care sunt in primul set, dar ia info din al doilea daca exista match */
    /* Daca ID_Angajat e si in dept2, DeptSalariu si Salariu vor fi disponibile */
    TITLE '6.2 Interclasare (MERGE) pe ID_Angajat';
RUN;
PROC PRINT DATA=angajati_merged; RUN;
TITLE;

/* Jonctiune SQL (PROC SQL) */
TITLE '6.3 Jonctiune INNER JOIN cu PROC SQL';
PROC SQL;
    CREATE TABLE angajati_sql_join AS
    SELECT a.ID_Angajat, a.Nume, a.Departament, b.Salariu
    FROM angajati_dept1 as a
    INNER JOIN angajati_dept2 as b
    ON a.ID_Angajat = b.ID_Angajat;
QUIT;
PROC PRINT DATA=angajati_sql_join; RUN;
TITLE;

TITLE '6.4 Jonctiune LEFT JOIN cu PROC SQL';
PROC SQL;
    CREATE TABLE angajati_sql_left_join AS
    SELECT a.ID_Angajat, a.Nume, a.Departament, b.Salariu, b.Departament as Departament_Dept2
    FROM angajati_dept1 as a
    LEFT JOIN angajati_dept2 as b
    ON a.ID_Angajat = b.ID_Angajat;
QUIT;
PROC PRINT DATA=angajati_sql_left_join; RUN;
TITLE;
"""
        run_sas_code(sas_code_combine, "Combinare Seturi de Date (SAS & SQL)")

        # 7. Utilizarea de masive (arrays)
        sas_code_arrays = """
/* 7. Utilizarea de masive (arrays) */
DATA vanzari_trimestriale;
    INPUT Produs $ T1 T2 T3 T4;
    ARRAY VanzariTrim T1 T2 T3 T4; /* Definire masiv explicit */
    TotalVanzari = 0;
    DO i = 1 TO DIM(VanzariTrim);
        TotalVanzari = TotalVanzari + VanzariTrim[i];
    END;
    MedieVanzari = MEAN(OF VanzariTrim[*]); /* Folosind OF cu masiv */
    DROP i;
    CARDS;
Laptop 10 12 15 13
Mouse 50 55 60 52
Tastatura 30 28 33 35
;
RUN;

PROC PRINT DATA=vanzari_trimestriale;
    TITLE '7. Utilizarea de masive pentru calcule';
RUN;
TITLE;

/* Transformare date din format lat in lung folosind masive */
DATA vanzari_lunare_lung;
    SET vanzari_trimestriale (KEEP=Produs T1 T2 T3 T4);
    ARRAY VanzariTrim {*} T1-T4; /* Masiv cu referinta la variabile existente */
    DO Trimestru = 1 TO DIM(VanzariTrim);
        ValoareVanzare = VanzariTrim[Trimestru];
        OUTPUT; /* Creeaza o noua observatie pentru fiecare trimestru */
    END;
    DROP T1-T4;
RUN;
PROC PRINT DATA=vanzari_lunare_lung;
    TITLE '7.1 Transformare lat-lung cu masive';
RUN;
TITLE;
"""
        run_sas_code(sas_code_arrays, "Utilizare Masive (Arrays)")
        
        # 8. Utilizarea de proceduri pentru raportare (PROC PRINT, PROC REPORT)
        sas_code_reporting = """
/* 8. Utilizarea de proceduri pentru raportare */
/* Set de date exemplu (folosim cel de la masive) */
/* PROC PRINT a fost deja folosit extensiv */

TITLE '8.1 Exemplu PROC REPORT simplu';
PROC REPORT DATA=vanzari_trimestriale NOWD; /* NOWD - No ODS Windowing Destination */
    COLUMN Produs T1 T2 T3 T4 TotalVanzari MedieVanzari;
    DEFINE Produs / DISPLAY "Nume Produs";
    DEFINE T1 / ANALYSIS SUM "Trim 1"; /* Specifica ca e variabila de analiza si eticheta */
    DEFINE T2 / ANALYSIS SUM "Trim 2";
    DEFINE T3 / ANALYSIS SUM "Trim 3";
    DEFINE T4 / ANALYSIS SUM "Trim 4";
    DEFINE TotalVanzari / ANALYSIS SUM "Total Anual";
    DEFINE MedieVanzari / ANALYSIS MEAN "Medie Trimestriala";
RUN;
TITLE;

TITLE '8.2 Exemplu PROC REPORT cu grupare si sumarizare';
PROC SORT DATA=vanzari_lunare_lung OUT=vanzari_sortate_report; BY Produs; RUN;
PROC REPORT DATA=vanzari_sortate_report NOWD;
    COLUMN Produs Trimestru ValoareVanzare;
    DEFINE Produs / GROUP "Produs"; /* Variabila de grupare */
    DEFINE Trimestru / ACROSS "Trimestru"; /* Variabila ACROSS pentru tabel incrucisat */
    DEFINE ValoareVanzare / ANALYSIS SUM "Vanzari";
    BREAK AFTER Produs / SUMMARIZE STYLE=[font_weight=bold]; /* Sumar dupa fiecare produs */
    RBREAK AFTER / SUMMARIZE STYLE=[font_weight=bold background=lightgray]; /* Sumar general */
RUN;
TITLE;
"""
        run_sas_code(sas_code_reporting, "Proceduri de Raportare (PRINT, REPORT)")
        
        # 9. Folosirea de proceduri statistice (PROC MEANS, PROC FREQ, PROC UNIVARIATE)
        sas_code_stats = """
/* 9. Folosirea de proceduri statistice */
/* Folosim setul de date studenti_info creat anterior */
TITLE '9.1 PROC MEANS - Statistici Descriptive';
PROC MEANS DATA=studenti_info MEAN MEDIAN MIN MAX N NMISS;
    VAR Varsta;
    CLASS Grupa; /* Statistici pe grupe */
RUN;
TITLE;

TITLE '9.2 PROC FREQ - Tabele de Frecventa';
PROC FREQ DATA=studenti_info;
    TABLES Grupa Varsta Grupa*Varsta / NOPERCENT NOCOL NOROW; /* Frecvente simple si incrucisate */
RUN;
TITLE;

TITLE '9.3 PROC UNIVARIATE - Analiza Detaliata a Variabilei';
PROC UNIVARIATE DATA=studenti_info NORMAL PLOT;
    VAR Varsta;
    HISTOGRAM Varsta / NORMAL; /* Histograma cu curba normala suprapusa */
    INSET MEAN STDDEV MIN MAX / POS=NE; /* Adauga statistici direct pe grafic */
RUN;
TITLE;
"""
        run_sas_code(sas_code_stats, "Proceduri Statistice (MEANS, FREQ, UNIVARIATE)")

        # 10. Generarea de grafice (PROC SGPLOT)
        sas_code_sgplot = """
/* 10. Generarea de grafice (PROC SGPLOT) */
/* Folosim setul de date studenti_info */

ODS GRAPHICS ON; /* Necesara pentru SGPLOT */

TITLE '10.1 Grafic de Bare Verticale (VBAR) - Numar Studenti pe Grupa';
PROC SGPLOT DATA=studenti_info;
    VBAR Grupa / RESPONSE=Varsta STAT=MEAN DATALABEL;
    XAXIS LABEL="Grupa";
    YAXIS LABEL="Varsta Medie";
RUN;
TITLE;

TITLE '10.2 Scatter Plot - Varsta (simulat vs. index)';
/* Cream o variabila index pentru scatter plot */
DATA studenti_plot;
    SET studenti_info;
    Index = _N_; /* Numarul observatiei */
RUN;
PROC SGPLOT DATA=studenti_plot;
    SCATTER X=Index Y=Varsta / GROUP=Grupa;
    XAXIS LABEL="Index Observatie";
    YAXIS LABEL="Varsta";
    KEYLEGEND / TITLE="Grupa";
RUN;
TITLE;

TITLE '10.3 Histograma Varstelor';
PROC SGPLOT DATA=studenti_info;
    HISTOGRAM Varsta / BINWIDTH=1 SHOWBINS;
    DENSITY Varsta / TYPE=NORMAL; /* Suprapune curba de densitate normala */
    DENSITY Varsta / TYPE=KERNEL; /* Suprapune estimare densitate kernel */
    XAXIS LABEL="Varsta";
RUN;
TITLE;
ODS GRAPHICS OFF;
"""
        run_sas_code(sas_code_sgplot, "Generare Grafice (SGPLOT)")

        # Placeholder for SAS ML if feasible later
        # st.subheader("11. SAS Machine Learning (Exemplu Simplu)")
        # st.info("Exemplele SAS ML cu `saspy` pot necesita module SAS/STAT sau SAS Viya specifice.")

    except saspy.sasexceptions.SASIOConnectionError as e_conn:
        st.error(f"Eroare de Conexiune SAS I/O la inițializarea paginii: {e_conn}")
        st.error("Verificați dacă serverul SAS este pornit și accesibil și dacă configurația saspy (ex: `sascfg_personal.py`) este corectă.")
        st.markdown("**Cauze comune:** Problema cu calea către executabilul SAS în `sascfg_personal.py`, SAS licențiat nefuncțional, serviciu SAS (remote/VM) nepornit.")
        if 'sas_session_demo' in st.session_state: st.session_state.sas_session_demo = None
    except saspy.sasexceptions.SASConfigNotFoundError as e_cfg:
        st.error(f"Eroare Critică de Configurare saspy la inițializarea paginii: {e_cfg}")
        st.error("Nu s-a găsit fișierul de configurare saspy (`sascfg.py` sau `sascfg_personal.py`) sau configurația specificată.")
        st.markdown("Vă rugăm creați sau verificați fișierul `sascfg_personal.py` în directorul home sau într-un loc accesibil Python.")
        if 'sas_session_demo' in st.session_state: st.session_state.sas_session_demo = None
    except Exception as e_page_init:
        st.error(f"Eroare Generală la inițializarea sesiunii SAS pentru pagina de demonstrații: {e_page_init}")
        st.error("Asigurați-vă că SAS este instalat și configurat corect pentru `saspy`. Verificați consola pentru detalii.")


# Ensure this is the last elif or handle the case where section doesn't match any known
# For example, if SAS_AVAILABLE is False, and user somehow lands on "SAS Runner"
# else:
#     st.error("Secțiune invalidă selectată.") 


# ... (any remaining code in app.py, if applicable)