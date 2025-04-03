# pages/3_Visualizations.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data

st.title("Vizualizări Educaționale")
df = load_data()

# 1. Grafic evolutie niveluri educatie
st.subheader("Evoluția numărului de elevi pe niveluri educaționale (1997-2023)")
fig1, ax1 = plt.subplots(figsize=(12, 6))
trend_data = df.groupby(['An', 'Nivel_Educatie'])['Valoare'].sum().reset_index()
sns.lineplot(data=trend_data, x='An', y='Valoare', hue='Nivel_Educatie', palette='tab10', ax=ax1)
ax1.set(title='Evoluția numărului de elevi pe nivel educațional',
       xlabel='An', ylabel='Număr elevi')
ax1.legend(title='Nivel Educațional', bbox_to_anchor=(1.05, 1))
ax1.grid(True)
st.pyplot(fig1)
st.write("""
Învățământul liceal înregistrează creștere constantă din 2010, pe când cel profesional scade. 
Vârfurile din 2005 și 2020 reflectă schimbări demografice și reforme educaționale. 
Disparitățile accentuate necesită revalorizarea învățământului profesional.""")

# 2. Distributie regionala
st.subheader("Distribuția elevilor pe regiuni (2023)")
total_by_region_2023 = df[df['An'] == 2023].groupby('Regiune')['Valoare'].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(12, 6))
total_by_region_2023.plot(kind='bar', color='skyblue', ax=ax2)
ax2.set(title='Distribuția pe regiuni', xlabel='Regiune', ylabel='Număr elevi')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig2)
st.write("""
București-Ilfov domină datorită concentrării resurselor educaționale. 
Regiunile vestice (Vest, Sud-Vest Oltenia) au cele mai scăzute valori. 
Disproporțiile reflectă dezvoltarea economică inegală a țării.""")

# 3. Distributie limbi predare
st.subheader("Distribuția elevilor pe limbă de predare (2023)")
total_by_language_2023 = df[df['An'] == 2023].groupby('Limba_Predare')['Valoare'].sum().sort_values(ascending=False)
fig3, ax3 = plt.subplots(figsize=(8, 8))
total_by_language_2023.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                           colors=sns.color_palette('pastel'), ax=ax3)
ax3.set(ylabel='', title='Limbile de predare')
st.pyplot(fig3)
st.write("""
Limba română predomina (83.7%), urmată de maghiară (12.1%) în zonele cu minorități. 
Limbi străine (germană 2.1%, engleză 1.8%) au pondere redusă. 
Diversitatea lingvistică rămâne limitată, cu potențial neexploatat.""")

# 4. Heatmap liceal
st.subheader("Heatmap: Învățământ liceal pe regiuni și ani (1997-2023)")
liceal_data = df[df['Nivel_Educatie'] == 'Invatamant liceal']
heatmap_data = liceal_data.pivot_table(index='An', columns='Regiune', values='Valoare', aggfunc='sum')
fig4, ax4 = plt.subplots(figsize=(14, 8))
sns.heatmap(heatmap_data.T, cmap='YlGnBu', annot=True, fmt='.0f',
           linewidths=.5, cbar_kws={'label': 'Număr elevi'}, ax=ax4)
ax4.set(title='Evoluția pe regiuni', xlabel='An', ylabel='Regiune')
st.pyplot(fig4)
st.write("""
București-Ilfov menține supremația cu valori crescânde după 2010. 
Scăderea generală din 2008-2012 corespunde crizei economice. 
Regiunile rurale (Sud-Vest, Nord-Est) rămân în urmă constant.""")

# 5. Comparatie limbi Nord-Vest
st.subheader("Comparație limbi de predare în Nord-Vest (1997-2023)")
nord_vest_data = df[df['Regiune'].str.upper() == 'NORD-VEST']
language_trend_nv = nord_vest_data.groupby(['An', 'Limba_Predare'])['Valoare'].sum().unstack()

if not language_trend_nv.empty:
    fig5, ax5 = plt.subplots(figsize=(14, 7))
    language_trend_nv.plot(kind='area', stacked=True, alpha=0.8, colormap='viridis', ax=ax5)
    ax5.set(title='Evoluția limbilor de predare în Nord-Vest',
           xlabel='An', ylabel='Număr elevi')
    ax5.legend(title='Limbă predare', bbox_to_anchor=(1.05, 1))
    ax5.grid(True)
    st.pyplot(fig5)
    st.write("""
    Predarea în română crește constant, absorbind ponderea limbilor minoritare. 
    Maghiara menține o prezență stabilă dar în scădere progresivă. 
    Limbi străine rămân marginale, deși regiunea este la granița UE.""")
else:
    st.write("Nu există date disponibile pentru Regiunea Nord-Vest.")