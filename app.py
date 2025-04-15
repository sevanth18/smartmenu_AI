import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

st.set_page_config(page_title="SmartMenu AI Assistant", layout="wide",)
st.title("🍲 SmartMenu: AI Assistant for Reducing Food Waste")
st.markdown("""
This assistant helps restaurant managers and chefs make smarter menu decisions.
Upload your restaurant data to get started.

""")

def get_base64_img(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_from_local(img_path):
    encoded = get_base64_img(img_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
set_bg_from_local("C:/Users/Sevanth Kumar/Downloads/download.jpg")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV File with Dish Data", type=["csv"])
def extract_waste(waste_str):
    try:
        ingredient, percent = waste_str.split('-')
        return ingredient.strip(), float(percent.strip().replace('%', '').replace('unused', ''))
    except:
        return None, 0.0



# --- Sample Logic After Upload ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df)
    print(df.columns)
    df.columns = df.columns.str.strip()
    if 'Ingredient Waste' in df.columns:
        df[['Waste Ingredient', 'Waste %']] = df['Ingredient Waste'].apply(lambda x: pd.Series(extract_waste(x)))
    else:
        st.error("Column 'Ingredient Waste' not found in the uploaded data.")


    # --- Preprocessing Waste Percent ---
    def extract_waste(row):
        try:
            percent = int(row.split('-')[1].strip().replace('% unused', '').replace('%', ''))
            ingredient = row.split('-')[0].strip()
            return ingredient, percent
        except:
            return None, 0

    df[['Waste Ingredient', 'Waste %']] = df['Ingredient Waste'].apply(lambda x: pd.Series(extract_waste(x)))


    # --- Insights Section ---
    st.subheader("🔫 Key Insights")
    most_wasted = df.sort_values(by='Waste %', ascending=False).iloc[0]
    st.markdown(f"- **Most Wasted Ingredient:** {most_wasted['Waste Ingredient']} ({most_wasted['Waste %']}%) from *{most_wasted['Dish Name']}*")

    low_sales = df[df['Weekly Orders'] < 5]
    if not low_sales.empty:
        st.markdown("- **Low-Selling Dishes (Consider Removal):**")
        st.write(low_sales[['Dish Name', 'Weekly Orders']])
    else:
        st.markdown("- All dishes have decent order volumes this week.")

    # --- Ingredient Reuse Suggestion ---
    st.subheader("🧪 Suggestions Using Wasted Ingredients")
    high_waste_ingredients = df[df['Waste %'] > 20]['Waste Ingredient'].unique()
    if len(high_waste_ingredients):
        st.markdown("You can try creating new dishes using these high-waste ingredients:")
        for ing in high_waste_ingredients:
            st.markdown(f"- **{ing}**: Try recipes like {ing} salad, {ing} dip, or {ing} smoothie")
    else:
        st.markdown("No high-waste ingredients found this week.")

    # --- Ingredient Overlap (High Margin Potential) ---
    st.subheader("💡 Dishes with Ingredient Overlap")
    ingredient_df = df[['Dish Name', 'Ingredients']].copy()
    ingredient_df['Ingredient List'] = ingredient_df['Ingredients'].apply(lambda x: [i.strip() for i in x.split(',')])

    overlap = {}
    dishes = ingredient_df.to_dict('records')
    for i in range(len(dishes)):
        for j in range(i+1, len(dishes)):
            common = set(dishes[i]['Ingredient List']).intersection(set(dishes[j]['Ingredient List']))
            if len(common) >= 2:
                key = f"{dishes[i]['Dish Name']} & {dishes[j]['Dish Name']}"
                overlap[key] = list(common)

    if overlap:
        for pair, common in overlap.items():
            st.markdown(f"**{pair}** share ingredients: {', '.join(common)}")
    else:
        st.markdown("No overlapping ingredient pairs with at least 2 shared items.")

    
    st.subheader("📊 Visual Dashboard: Weekly Orders, Waste % & Prep Time")

    # Preprocessing
    df['Prep Time (min)'] = df['Avg Prep Time'].str.replace('mins', '').str.strip().astype(int)
    df['Ingredient Cost (₹)'] = df['Ingredient Cost'].replace({'₹': ''}, regex=True).astype(float)
    df['Waste %'] = df['Ingredient Waste'].apply(lambda x: float(x.split('-')[1].replace('% unused', '').strip()))

    # Top 5 datasets
    top_orders = df.sort_values(by='Weekly Orders', ascending=False).head(5)
    top_waste = df.sort_values(by='Waste %', ascending=False).head(5)
    top_prep = df.sort_values(by='Prep Time (min)', ascending=False).head(5)

    # --- Row 1: Two Pie Charts Side by Side ---
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(4.5, 4.5))
        ax1.pie(
            top_orders['Weekly Orders'],
            labels=top_orders['Dish Name'],
            autopct='%1.1f%%',
            startangle=140,
            textprops={'fontsize': 8},
            colors=plt.cm.Blues(np.linspace(0.3, 0.85, 5))
            
        )
        ax1.set_title("Top 5 Dishes by Weekly Orders", fontsize=10)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
        ax2.pie(
            top_waste['Waste %'],
            labels=top_waste['Dish Name'],
            autopct='%1.1f%%',
            startangle=140,
            textprops={'fontsize': 8},
            colors=plt.cm.Reds(np.linspace(0.3, 0.85, 5))
        )
        ax2.set_title("Top 5 Dishes by Waste %", fontsize=10)
        st.pyplot(fig2)

    # --- Row 2: Horizontal Bar Chart (Centered) ---
    st.markdown("### 🕒 Top 5 Dishes by Prep Time")
    center_col = st.columns([1, 3, 1])[1]  # Middle column for centering

    with center_col:
        fig3, ax3 = plt.subplots(figsize=(5, 4.5))
        colors=plt.cm.Paired(np.linspace(0, 1, len(top_prep)))
        ax3.barh(top_prep['Dish Name'], top_prep['Prep Time (min)'], color=colors)
        for i, (val, name) in enumerate(zip(top_prep['Prep Time (min)'], top_prep['Dish Name'])):
            ax3.text(val + 0.5, i, f"{val} min", va='center', fontsize=8)
        ax3.set_xlabel("Prep Time (min)")
        ax3.set_title("Longest Avg Prep Time Dishes", fontsize=10)
        ax3.invert_yaxis()
        st.pyplot(fig3)


    st.markdown("## 🤖 Smart Suggestions from AI Assistant")

    # 1. Most wasted ingredient
    most_wasted_ing = df.sort_values(by="Waste %", ascending=False).iloc[0]
    st.markdown(f"**🟠 Most Wasted Ingredient:** {most_wasted_ing['Waste Ingredient']} – {most_wasted_ing['Waste %']}% unused")

    # 2. Suggest a new dish using ingredients we already have
    all_ingredients = set(i.strip() for sublist in df['Ingredients'].str.split(',') for i in sublist)
    common_dish_suggestions = [
        f"{ing} smoothie" for ing in all_ingredients if ing.lower() in ['banana', 'mango', 'avocado']
    ] + [
        f"{ing} soup" for ing in all_ingredients if ing.lower() in ['tomato', 'onion', 'lemon']
    ]
    if common_dish_suggestions:
        st.markdown("**🧪 Suggested New Dish Ideas Using Current Ingredients:**")
        for d in common_dish_suggestions:
            st.markdown(f"- {d.title()}")
    else:
        st.markdown("**🧪 No dish suggestions found with existing ingredients.**")

    # 3. Dishes both low-selling and high-waste
    low_sell_high_waste = df[(df['Weekly Orders'] < 5) & (df['Waste %'] > 20)]
    if not low_sell_high_waste.empty:
        st.markdown("**🔻 Dishes to Consider Removing (Low-Selling & High-Waste):**")
        st.dataframe(low_sell_high_waste[['Dish Name', 'Weekly Orders', 'Waste %']])
    else:
        st.markdown("**✅ No dishes met the criteria for low-selling and high-waste this week.**")

    # 4. What to stock less of next week
    stock_less = df[df['Waste %'] > 25]['Waste Ingredient'].value_counts()
    if not stock_less.empty:
        st.markdown("**📉 Ingredients to Stock Less of Next Week (High Waste):**")
        for ing, count in stock_less.items():
            st.markdown(f"- {ing} (high waste in {count} dish{'es' if count > 1 else ''})")
    else:
        st.markdown("**👍 All ingredients are being utilized efficiently.**")



else:
    st.info("Please upload a CSV file to begin analysis.")




st.markdown("---")
st.caption("Built with ❤️ using Streamlit",)