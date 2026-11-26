# Experiment 1: Data Analysis

**Name:** Lakshya Upadhyay  
**SAP ID:** 500119462  
**Specialisation Batch:** AI/ML B1  
**Subject:** Computational Linguistics & Natural Language Processing  

## Objective
To perform exploratory data analysis (EDA) on structured employee records to uncover demographic, salary, and experience distributions using Python data manipulation and visualization libraries (`pandas`, `matplotlib`, `seaborn`).

## Procedure & Implementation
The experiment operates on `employee_information_100.csv` containing 100 employee entries across 7 attributes (`Employee_ID`, `Name`, `Department`, `Age`, `Gender`, `Salary`, `Experience_Years`). The workflow executes the following analytical tasks:

1. **Dataset Inspection (`df.info()`, `df.head()`):**
   * Inspect column data types, memory usage, and check for missing/null values across all attributes.

2. **Average Salary by Department (`df.groupby('Department')['Salary'].mean()`):**
   * Group records by department to compute mean salaries and visualize with a horizontal bar chart (`plot(kind="barh")`).

3. **Employee Count per Department (`df['Department'].value_counts()`):**
   * Compute department-wise employee frequencies and visualize the distribution with a Seaborn count plot (`sns.countplot()`).

4. **Gender Demographics (`df['Gender'].value_counts()`):**
   * Aggregate workforce gender frequencies and display proportional distribution using a Matplotlib pie chart (`plt.pie()`).

5. **Salary Distribution Analysis (`sns.histplot(kde=True)`):**
   * Plot a histogram with Kernel Density Estimation (15 bins) to examine salary skewness, modal salary brackets, and variance.

6. **Experience vs. Salary Correlation (`sns.scatterplot()`):**
   * Generate a multi-variable scatter plot comparing `Experience_Years` against `Salary`, utilizing `hue='Department'` and `style='Gender'` to evaluate seniority and compensation trends.

7. **Top 10 Highest Earners (`df.nlargest(10, 'Salary')`):**
   * Extract and display details of the top 10 highest-paid employees across `Name`, `Department`, `Salary`, and `Experience_Years`.

8. **Departmental Maximum Salaries (`df.groupby('Department')['Salary'].max()`):**
   * Determine peak compensation figures for each individual department.

9. **Above-Average Earners (`df[df['Salary'] > df['Salary'].mean()]`):**
   * Filter records where employee salary strictly exceeds the organizational mean salary, recording both total headcount and sample profiles.

10. **Average Experience per Department (`df.groupby('Department')['Experience_Years'].mean()`):**
    * Calculate average employee tenure/experience per department to evaluate departmental seniority.

11. **Age Distribution Analysis (`sns.histplot(kde=True)`):**
    * Plot a histogram with KDE curve (15 bins) to visualize the age demographic spread of the workforce.

## Observations
| Analysis Task | Key Observation Parameter | Implementation / Method | Result Output |
| :--- | :--- | :--- | :--- |
| **Data Inspection** | 100 rows, 7 columns, non-null | `df.info()`, `df.head()` | Verified clean tabular data |
| **Departmental Salary** | Mean salary variance across departments | `groupby('Department')['Salary'].mean()` | Sorted Horizontal Bar Chart |
| **Employee Distribution** | Workforce concentration per department | `df['Department'].value_counts()` | Seaborn Count Plot |
| **Gender Demographics** | Proportion of male vs female employees | `df['Gender'].value_counts()` | Matplotlib Pie Chart |
| **Salary Spread** | Salary distribution and density curve | `sns.histplot(kde=True, bins=15)` | Histogram with KDE |
| **Experience & Pay** | Correlation between years worked and salary | `sns.scatterplot(hue='Department', style='Gender')` | Multi-category Scatter Plot |
| **Top Earners** | Top 10 highest-paid personnel | `df.nlargest(10, 'Salary')` | Filtered DataFrame table |
| **Maximum Salaries** | Peak compensation limits per department | `groupby('Department')['Salary'].max()` | Aggregated Series table |
| **Above-Average Base** | Count of employees exceeding mean salary | `df[df['Salary'] > mean()]` | Boolean Masked DataFrame |
| **Experience Trends** | Average seniority levels by department | `groupby('Department')['Experience_Years'].mean()` | Aggregated Series table |
| **Age Spread** | Age demographic distribution | `sns.histplot(df['Age'], kde=True, bins=15)` | Histogram with KDE |

## Result Interpretation & Learnings
* **Holistic EDA Mastery:** Demonstrated the end-to-end exploratory workflow on structured tabular data, combining statistical aggregation with descriptive visualization.
* **Compensation Insights:** Confirmed a positive correlation between years of experience and salary across departments, while identifying department-level salary baselines and maximum caps.
* **Library Synergy:** Leveraged Pandas for aggregation/filtering, Matplotlib for customized figure geometry and pie charts, and Seaborn for statistical distributions and multi-dimensional scatter plots.
