import warnings

import matplotlib.pyplot as plt
from waterfall_chart import plot

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Cambria'

# not using `iris`
categories = ['sales', 'returns', 'credit fees', 'rebates', 'late charges', 'shipping']
values = [335, -208, -170.5, -250, 943, -100]
waterfall_plot = plot(categories, values)

plt.title('Waterfall Chart', pad=20)
plt.xlabel('Transactions', labelpad=15)
plt.ylabel('Amount in K', labelpad=15)
plt.margins(x=0.05, y=0.1)

# plt.show()

plt.savefig('plots/waterfall_plot.png', dpi=300, bbox_inches='tight')
plt.close()
