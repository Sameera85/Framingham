#import libraries
import streamlit as st  # for creating web apps
import numpy as np  # for numerical computing
import pandas as pd # for dataframe and manipulation
import seaborn as sns #for graphs and data visualization
from matplotlib import pyplot as plt 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import plotly.express as px #for graphs and data visualization
sns.set() #setting seaborn as default for plots


# CSS Styling
# Define the path of CSS file
css_path = 'style.css'

# Load the contents of CSS file
with open(css_path) as f:
    css = f.read()

# Use the st.markdown function to apply the CSS to Streamlit app
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


# load the dataset
df = pd.read_csv('framingham_clean.csv')

# create a sidebar for the project

with st.sidebar:

    st.markdown('<h1 class="sidebar-title">Coronary Heart Disease</h1>', unsafe_allow_html=True)
    st.image('./heart.jpeg' , caption="CHD")

    # Create a collapsible container for the project overview
    with st.expander("Overview"):
        st.write("""
    This dataset contains information related to cardiovascular disease risk factors for a group of individuals. Cardiovascular diseases, including heart disease and stroke, are significant health concerns worldwide. Understanding the risk factors associated with these diseases is essential for prevention and management.

    """) 
    
# Create the expander using the custom styling from the external CSS
st.expander("Custom Styled Expander", expanded=False)


# Create a container div with the specified CSS class
#st.markdown('<div class="my-container">', unsafe_allow_html=True)

# Set Title of project with a CSS class
st.markdown('<h1 class="my-title ">Framingham Heart Study</h1>', unsafe_allow_html=True)

## 4.EDA Questions

### 4.1"What is the distribution of age among the dataset, and how does it relate to the gender distribution? 

st.markdown('<h3 class="sub-header">1- Distribution of Age & Gender among the dataset. </h3>', unsafe_allow_html=True)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Create a histogram for the age distribution
most_frequent_age = df['age'].mode().values[0]
sns.histplot(df['age'], bins=20, kde=True, ax=ax1)
ax1.set_title('Age Distribution (Most Frequent Age)')
ax1.set(xlabel='Age', ylabel='Frequency')
ax1.annotate(f'Most Frequent Age: {most_frequent_age}', xy=(most_frequent_age, 0), xytext=(most_frequent_age, 50),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'), color='red')

# Create a function to plot the gender distribution
def plot_gender_distribution():
    sns.countplot(data=df, x='gender', ax=ax2)
    ax2.set_title('Gender Distribution')
    ax2.set_xlabel('Gender')
    ax2.set_ylabel('Count')
    total_count = len(df)
    for p in ax2.patches:
        percentage = f'{100 * p.get_height() / total_count:.1f}%'
        ax2.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    ax2.set_xticklabels(['Female', 'Male'])

# Call the function to plot the gender distribution
plot_gender_distribution()

# Display the plot in Streamlit
st.pyplot(fig)

### 4.2 "What is the distribution of key cardiovascular risk factors in the dataset, and how does their skewness impact the understanding of the data's central tendency and spread? 

with st.expander(" 2. Distribution of key Cardiovascular Risk Factors with Skewness Labels"):
    # Define colors for each variable
    colors = ['orange', 'green', 'red', 'purple', 'blue', 'grey']

    # Create a figure to contain all the subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    variables = ['glucose', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate']

    # Define skewness labels based on your criteria
    def get_skew_label(skewness):
        if skewness < -1 or skewness > 1:
            return 'Highly Skewed'
        elif (-1 <= skewness <= -0.5) or (0.5 <= skewness <= 1):
            return 'Moderately Skewed'
        else:
            return 'Approximately Symmetric'

    for i, var in enumerate(variables):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        # Calculate skewness
        skewness = df[var].skew()
        skew_label = get_skew_label(skewness)

        sns.histplot(df[var], color=colors[i], kde=True, ax=ax)
        ax.set_title(f'Distribution of {var}\nSkewness: {skewness:.2f} ({skew_label})')

    # Display the entire figure in Streamlit
    st.pyplot(fig)

### 4.3 "How does the distribution of heart rate and age groups in the dataset provide insights into the demographics and health characteristics of the surveyed population, and is there any noticeable correlation between these factors? 

# Define the columns for the Streamlit app
col1, col2 = st.columns(2)

# Create a DataFrame for Plotly
plotly_df = df.copy()
plotly_df['heart_rate_groups'] = plotly_df['heart_rate_groups'].map({0: 'Low', 1: 'Normal', 2: 'High'})
plotly_df['age_groups'] = plotly_df['age_groups'].map({0: 'Adults', 1: 'Middle-Aged', 2: 'Senior'})

st.markdown('<P class="sub-header">2. Heart Rate and Age Group Analysis </p>', unsafe_allow_html=True)
# Create subplots
fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Count by HeartRate Group', 'Count by Age Group'))

# Create the first plot (HeartRate Grouped)
fig3.add_trace(go.Bar(x=plotly_df['heart_rate_groups'].value_counts().index, y=plotly_df['heart_rate_groups'].value_counts(), marker_color='lightcoral'), row=1, col=1)
fig3.update_xaxes(title_text='Heart Rate Group', row=1, col=1)
fig3.update_yaxes(title_text='Count', row=1, col=1)

# Create the second plot (Count by Age Group)
fig3.add_trace(go.Bar(x=plotly_df['age_groups'].value_counts().index, y=plotly_df['age_groups'].value_counts(), marker_color='lightblue'), row=1, col=2)
fig3.update_xaxes(title_text='Age Group', row=1, col=2)
st.plotly_chart(fig3)

### 4.4 Is there a difference in the number of male and female patients with coronary heart disease? 
### 4.5 How does the prevalence of diabetes vary across different age groups, and what percentage of people in each age group have diabetes? 


st.markdown('<h3 class="sub-header">3. Diabetes and Coronary Heart Disease by Age Group and Gender</h2>', unsafe_allow_html=True)
# Define age group labels
age_group_labels = ['Adults', 'Middle-Aged', 'Senior']

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# First subplot: Diabetes by Age Group
plt.sca(axes[0])  # Set the current axes to the first subplot
ax1 = sns.countplot(x='age_groups', hue='diabetes', data=df, palette='rainbow')
plt.xlabel('Age Group')
plt.ylabel('No. of Patients')
plt.xticks(ticks=[0, 1, 2], labels=age_group_labels)
plt.legend(title='Diabetes', labels=['Negative', 'Positive'])
plt.title('Diabetes by Age Group')
total_count1 = len(df)
for p in ax1.patches:
    height = p.get_height()
    percentage = height / total_count1 * 100
    ax1.text(p.get_x() + p.get_width() / 2, height + 5, f'{percentage:.1f}%', ha='center')

# Second subplot: Coronary Heart Disease by Gender
plt.sca(axes[1])  # Set the current axes to the second subplot
sns.set_style("whitegrid")  # Add grid lines
ax2 = sns.countplot(x='gender', hue='TenYearCHD', data=df, palette='Paired')
plt.xlabel('Gender', fontsize=14)
plt.xticks(ticks=[0, 1], labels=['Female', 'Male'], fontsize=12)
plt.ylabel('No. of Patients', fontsize=14)
plt.yticks(fontsize=12)
plt.legend(['Neg.', 'Pos.'], title='CHD Status', title_fontsize=12, fontsize=12)
plt.title('Coronary Heart Disease (CHD) by Gender', fontsize=16)
total_count2 = len(df)
for p in ax2.patches:
    height = p.get_height()
    percentage = height / total_count2 * 100
    ax2.text(p.get_x() + p.get_width() / 2, height + 5, f'{percentage:.1f}%', ha='center')

# Adjust plot aesthetics
sns.despine(left=True, ax=axes[1])
axes[1].set_axisbelow(True)  # Move grid lines behind the bars

# Display the subplots in Streamlit
st.pyplot(fig)

### 4.6 "How do systolic and diastolic blood pressures vary across different age groups and genders, and are there any noticeable patterns or differences that could indicate potential health trends or risk factors? 
### 4.7 How do glucose and total cholesterol vary across different age groups and genders, and are there any noticeable patterns or differences that could indicate potential health trends or risk factors? 

# Create age group labels
age_group_labels = ['Adults', 'Middle-Aged', 'Senior']

# Box plots for Sys. BP by Age Group & Gender
fig1 = px.box(df, x='age_groups', y='sysBP', color='gender', title='Sys. BP vs Age Group by Gender')
fig1.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

# Boxen plots for Dia. BP by Age Group & Gender
fig2 = px.box(df, x='age_groups', y='diaBP', color='gender', title='Dia. BP vs Age Group by Gender')
fig2.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

# Box plots for Sys. BP by Age Group & Gender
fig3 = px.box(df, x='age_groups', y='glucose', color='gender', title='Glucose vs Age Group by Gender')
fig3.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

# Boxen plots for Dia. BP by Age Group & Gender
fig4 = px.box(df, x='age_groups', y='totChol', color='gender', title='Total Cholesterol vs Age Group by Gender')
fig4.update_xaxes(categoryorder='array', categoryarray=age_group_labels)

# Create a checkbox in the sidebar for selecting the figure to display
selected_figure = st.sidebar.radio("Health Metrics Analysis by Age Group and Gender", [None,'Sys. BP', 'Dia. BP', 'Glucose', 'Total Cholesterol'])

# Display the selected figure in the main area
if selected_figure == 'Sys. BP':
    st.plotly_chart(fig1)
elif selected_figure == 'Dia. BP':
    st.plotly_chart(fig2)
elif selected_figure == 'Glucose':
    st.plotly_chart(fig3)
elif selected_figure == 'Total Cholesterol':
    st.plotly_chart(fig4)

### 4.8 How does the number of cigarettes smoked per day ('cigsPerDay') vary across different age groups? 

# Define age group labels
age_group_labels = ['Adults', 'Middle-Aged', 'Senior']

# Create a density plot for 'cigsPerDay' by age group
plt.figure(figsize=(10, 7))
sns.set(style="whitegrid")

# Create a list of colors
colors = ['turquoise', 'coral', 'gold']

# Create a custom color palette
palette = sns.color_palette(colors, as_cmap=True)

# Plot the density plot for 'cigsPerDay' by age group
sns.kdeplot(data=df, x='cigsPerDay', hue='age_groups', common_norm=False, fill=True, palette=palette)
st.markdown('<h3 class="sub-header"> 4- Cigs. per day by Age Group (Density Plot)</h3>', unsafe_allow_html=True)
plt.xlabel('Cigs. / Day')
plt.ylabel('Density')
# Display the plot in Streamlit
st.pyplot(plt.gcf())

### 4.9 How are systolic blood pressure, diastolic blood pressure, total cholesterol, and 10-year coronary heart disease risk related to each other?

# Sidebar Filters
st.sidebar.title("Scatter Plot")
age_group = st.sidebar.slider("Select Age Group", min_value=30, max_value=70, value=(30, 70))


# Apply filters to the data
filtered_data = df.copy()
# Add color variable based on gender and smoker status
if age_group[0] != 30 or age_group[1] != 70:
    filtered_data = filtered_data[(filtered_data["age"] >= age_group[0]) & (filtered_data["age"] <= age_group[1])]
# Scatter plot


# Select features from the predefined list
x_feature = st.sidebar.selectbox("X-axis Feature", ["totChol", "sysBP", "diaBP","heartRate", "glucose"])
y_feature = st.sidebar.selectbox("Y-axis Feature", ["glucose", "heartRate","diaBP", "sysBP", "totChol"])


# Select color variable
color_variable = st.sidebar.selectbox("Color Variable", ["None", "Gender", "Smoking Status", "Ten Year CHD"])
size_feature = "BMI"
if color_variable == "Gender":
    fig = px.scatter(
        filtered_data,
        x=x_feature,
        y=y_feature,
        size=size_feature,  # Specify the size variable
        color="gender",
        labels={x_feature: x_feature, y_feature: y_feature},
        title=f"{x_feature} vs. {y_feature} by Gender"
    )
elif color_variable == "Smoking Status":
    fig = px.scatter(
        filtered_data,
        x=x_feature,
        y=y_feature,
        color="currentSmoker",
        size=size_feature,  # Specify the size variable
        labels={x_feature: x_feature, y_feature: y_feature},
        title=f"{x_feature} vs. {y_feature} by Smoking Status"
    )
elif color_variable == "Ten Year CHD":
    fig = px.scatter(
        filtered_data,
        x=x_feature,
        y=y_feature,
        color="TenYearCHD",
        size=size_feature,  # Specify the size variable

        labels={x_feature: x_feature, y_feature: y_feature},
        title=f"{x_feature} vs. {y_feature} by Ten Year CHD"
       
    )
else:
    fig = px.scatter(
        filtered_data,
        x=x_feature,
        y=y_feature,
        size=size_feature,  # Specify the size variable
        labels={x_feature: x_feature, y_feature: y_feature},
        title=f"{x_feature} vs. {y_feature}"
    )

st.plotly_chart(fig)
st.sidebar.markdown("")
st.sidebar.markdown("<h4 style='color: blue; font-size: 16px;'>Made with 💙 Eng.Sameera alkhalifi</h4>", unsafe_allow_html=True)




# close the container div with the specified CSS class
st.markdown('</div>', unsafe_allow_html=True)



