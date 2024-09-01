import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.DataFrame(
    {'Complaint Type': ['Late Delivery', 'Product Quality', 'Customer Service', 'Billing Issue', 'Shipping Error'],
     'Count': [150, 120, 90, 60, 30]})
df_sorted = df.sort_values(by='Count', ascending=False)
df_sorted['Cumulative Percentage'] = df_sorted['Count'].cumsum() / df_sorted['Count'].sum() * 100
df_sorted['Count'] = df_sorted['Count'] / 2

fig, ax1 = plt.subplots(figsize=(12, 8), dpi=150)
color_palette = sns.color_palette("pastel", len(df_sorted))
ax2 = ax1.twinx()

bars = ax1.bar(df_sorted['Complaint Type'], df_sorted['Count'], color=color_palette, width=0.25)
ax1.set_xlabel('Complaint Type', fontsize=14, fontname='Cambria')
ax1.set_ylabel('Count', color='tab:blue', fontsize=14, fontname='Cambria')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

ax2.set_ylabel('Cumulative Percentage (%)', color='tab:red', fontsize=14, fontname='Cambria')
line = ax2.plot(df_sorted['Complaint Type'], df_sorted['Cumulative Percentage'], color='tab:red', marker='o',
                linestyle='dashed')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 100)

for i in range(len(df_sorted)):
    ax1.text(i, df_sorted['Count'].iloc[i] - 5, df_sorted['Count'].iloc[i], ha='center', color='tab:blue', fontsize=12,
             fontname='Cambria')
    ax2.text(i, df_sorted['Cumulative Percentage'].iloc[i] - 5, f"{df_sorted['Cumulative Percentage'].iloc[i]:.1f}%",
             ha='center', color='tab:red', fontsize=12, fontname='Cambria')
plt.title('Pareto Chart of Customer Complaints', fontsize=16, fontname='Cambria')
fig.tight_layout()

# plt.show()

plt.savefig('plots/pareto_chart.png')
plt.close()
