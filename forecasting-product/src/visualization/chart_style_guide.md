# Chart Style Guide — Forecasting Platform

Guidelines for creating consistent, SWD-style charts across the platform.
All charts — whether static (matplotlib) or interactive (Plotly) — should
follow these principles.

---

## Principles

1. **Focus vs. context.** One element highlighted, everything else gray.
2. **Action titles.** Titles state the takeaway, not the chart type.
   - ✅ "LightGBM achieves lowest WMAPE across all series"
   - ❌ "Model Comparison Bar Chart"
3. **Direct labels.** Label data points directly. Remove legends when possible.
4. **Minimal chrome.** No top/right spines, no gridlines on categorical axis,
   no rotated text, no 3D effects.
5. **Color with purpose.** Color encodes meaning (good/bad/alert), not decoration.

---

## Color Palette

### Core Colors

| Name       | Hex       | Usage                                 |
|------------|-----------|---------------------------------------|
| Primary    | `#4361EE` | Main data series, links, actions      |
| Secondary  | `#3A0CA3` | Deep purple accent                    |
| Accent     | `#F72585` | Highlight anomalies, key findings     |
| Success    | `#06D6A0` | Positive signals, adds-value, champion|
| Warning    | `#FFD166` | Caution, neutral zones                |
| Danger     | `#EF476F` | Alerts, destroys-value, violations    |
| Neutral    | `#8D99AE` | Context, supporting elements          |

### Semantic Palettes

**FVA cascade:**
- Adds Value → `#06D6A0` (success green)
- Neutral → `#8D99AE` (gray)
- Destroys Value → `#EF476F` (danger red)
- Baseline → `#4361EE` (primary blue)

**Model layers:**
- Naive → `#8D99AE`
- Statistical → `#4361EE`
- ML → `#06D6A0`
- Neural → `#F72585`
- Foundation → `#3A0CA3`
- Intermittent → `#FFD166`
- Ensemble → `#7209B7`
- Override → `#FF6B35`

**Demand classes:**
- Smooth → `#06D6A0`
- Intermittent → `#FFD166`
- Erratic → `#F72585`
- Lumpy → `#EF476F`

### Categorical (for multi-series)
8 colorblind-safe colors in order:
`#4361EE`, `#F72585`, `#06D6A0`, `#FFD166`, `#3A0CA3`, `#FF6B35`, `#7209B7`, `#8D99AE`

---

## Chart Decision Tree

| Question                           | Chart Type           | Builder Function           |
|------------------------------------|----------------------|----------------------------|
| Compare models/categories?         | Horizontal bar       | `highlight_bar()`          |
| Show metric over time?             | Line                 | `highlight_line()`         |
| Show forecast with actuals?        | Forecast line        | `forecast_plot()`          |
| Monitor metric stability?          | Control chart        | `control_chart_plot()`     |
| Rank models by performance?        | Leaderboard bars     | `leaderboard_chart()`      |
| Show FVA by layer?                 | Cascade bars         | `fva_cascade_chart()`      |
| Track drift over time?             | Drift timeline       | `drift_timeline()`         |
| Show demand class distribution?    | Horizontal bars      | `demand_class_chart()`     |
| Compare two scenarios/periods?     | Fill-between lines   | `fill_between_lines()`     |
| Show composition/parts?            | Stacked bar          | `stacked_bar()`            |
| Show step-by-step drop-off?        | Funnel waterfall     | `funnel_waterfall()`       |
| Display KPI summary?               | Big number card      | `big_number_layout()`      |

---

## Declutter Checklist

Before saving any chart, verify:

- [ ] Top and right spines removed
- [ ] No gridlines on the categorical axis
- [ ] Y-axis gridlines are light gray (`#E5E7EB`) and thin (0.5px)
- [ ] Axis labels are gray (`#6B7280`), not black
- [ ] Tick marks are hidden (size = 0)
- [ ] Data labels present — legend used only when unavoidable
- [ ] Color used for meaning, not decoration
- [ ] Background is `#F8F9FA` (light warm gray)
- [ ] Title is action-oriented and left-aligned

---

## Anti-Patterns (Do Not Use)

- ❌ Rainbow color palettes
- ❌ 3D effects on any chart
- ❌ Rotated axis labels (restructure the chart instead)
- ❌ Pie charts (use horizontal bars)
- ❌ Dual y-axes (use two panels)
- ❌ Chart junk (borders, shadows, gradient fills)
- ❌ Red/green encoding without shape redundancy (colorblind issue)

---

## Usage Examples

### Matplotlib (static export / reports)

```python
from src.visualization.chart_helpers import (
    swd_style, highlight_bar, action_title, save_chart, CHART_FIGSIZE,
    forecast_plot, leaderboard_chart, fva_cascade_chart,
)
import matplotlib.pyplot as plt

# 1. Always call swd_style() first
swd_style()

# 2. Create chart
fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
highlight_bar(ax, model_names, wmape_scores, highlight="LightGBM")
action_title(ax, "LightGBM achieves lowest WMAPE", subtitle="Backtest results")

# 3. Save
save_chart(fig, "outputs/leaderboard.png")
```

### Plotly (Streamlit dashboard)

```python
from src.visualization.plotly_theme import apply_swd_plotly_theme, swd_plotly_layout
import plotly.express as px

# 1. Apply theme once at app startup
apply_swd_plotly_theme()

# 2. All subsequent charts inherit the style
fig = px.bar(df, x="model", y="wmape", color="fva_class")
fig.update_layout(title="FVA cascade — error by forecasting layer")

# 3. Or customize per-chart
fig.update_layout(**swd_plotly_layout(title={"text": "Custom title"}))
```

---

## WCAG Compliance

All foreground text colors meet WCAG 2.1 AA contrast (4.5:1) against the
`#F8F9FA` background. Use `ensure_contrast()` from `chart_palette.py` if
you introduce custom colors.
