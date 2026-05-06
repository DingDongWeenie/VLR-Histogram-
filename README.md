# VLR Histogram Analysis

This repository contains Python scripts for analyzing and visualizing professional VALORANT player performance metrics using statistical distributions and real data from the VLR (VALORANT Champions League) bronze dataset.

## Project Overview

These scripts demonstrate statistical analysis of esports data, focusing on how different probability distributions model player performance metrics in competitive VALORANT.

---

## VLR.py - Real Data Histogram

### What It Does
Analyzes real Average Combat Score (ACS) data from professional VALORANT players in the bronze dataset. It loads thousands of CSV files, deduplicates players, and creates a histogram visualization.

### Graph Explanation
**Histogram of Average Combat Score (ACS) Distribution**

- **X-Axis**: Average Combat Score (ACS) values
- **Y-Axis**: Number of players with that ACS score
- **Red Dashed Line**: Mean (average) ACS across all players
- **Green Dashed Line**: Median ACS across all players
- **Curve (KDE)**: Kernel Density Estimation showing the smooth probability distribution

### Key Statistics Generated
- **Total Players Analyzed**: Count of unique players in the dataset
- **Average ACS**: Mean performance score
- **Median ACS**: Middle value when all scores are sorted
- **Min/Max ACS**: Range of player performance
- **Standard Deviation**: How much individual scores vary from the mean

### What ACS Means
ACS (Average Combat Score) is a metric that measures a player's average contribution per round. Higher ACS indicates better overall gameplay impact including kills, assists, and economy management.

### Usage
```bash
python VLR.py bronze
python VLR.py bronze --output histogram.png
```

---

## VLR_distributions.py - Synthetic Distribution Analysis

### What It Does
Generates 5 different probability distributions using synthetic data based on realistic VALORANT player metrics. Creates a 3x2 grid showing different statistical distribution shapes.

### Graphs Explanation

#### 1. **Exponential Distribution - Time to First Elimination (seconds)**
- **Shape**: Steep curve that drops off quickly
- **What It Represents**: How quickly players get their first kill/elimination
- **Interpretation**: Most eliminations happen early in the round (~0-10 seconds). As time increases, fewer eliminations occur exponentially
- **Real-World Meaning**: Players who coordinate early plays get eliminations fast; scattered eliminations are rarer

#### 2. **Normal Distribution - Average Combat Score (ACS)**
- **Shape**: Classic bell curve centered around the mean
- **What It Represents**: How most players' performance scores cluster around an average
- **Interpretation**: Most professional players have similar skill levels, clustering around ~180 ACS. Outliers exist but are rare
- **Real-World Meaning**: The majority of pro players perform within a predictable range

#### 3. **Poisson Distribution - Kills Per Round (Individual Player)**
- **Shape**: Multiple peaks at whole numbers, concentrated on lower values
- **What It Represents**: Number of kills a single player gets per round
- **Interpretation**: Shows count-based data where 0-2 kills per round are most common, with fewer instances of 5+ kills
- **Real-World Meaning**: Players rarely get more than 2-3 kills per round; higher kill rounds are exceptional

#### 4. **Binomial Distribution - Headshot Rate (out of 10 shots)**
- **Shape**: Concentrated peak showing success rate
- **What It Represents**: Out of 10 shots, how many hit the head with ~65% accuracy
- **Interpretation**: Most players land 6-7 headshots per 10 shots. Some get fewer (misses), some get more (lucky streaks)
- **Real-World Meaning**: Weapon accuracy is fairly consistent around the expected success rate

#### 5. **Triangular Distribution - Ability Usage Frequency (%)**
- **Shape**: Triangle-like peak in the middle, decreasing toward edges
- **What It Represents**: How often players use their abilities during a match
- **Interpretation**: Players tend to use abilities 50-80% of the time. Underuse (<20%) and overuse (>90%) are both rare
- **Real-World Meaning**: Optimal ability usage is mid-range; too little or too much is suboptimal

### Statistics Generated for Each Distribution
- **Mean**: Average value across all samples
- **Median**: Middle value when sorted
- **Standard Deviation**: Spread/variance of the data
- **Skewness**: Asymmetry (-1 to +1 range, 0 = symmetric)
- **Kurtosis**: Tailedness (how extreme the outliers are)

### Usage
```bash
python VLR_distributions.py
python VLR_distributions.py --output distributions_chart.png
```

---

## Key Concepts

### What Are Probability Distributions?
Probability distributions describe how values are spread or distributed across a range. Different types of real-world data follow different patterns:
- **Normal**: Heights, test scores, many natural phenomena
- **Exponential**: Time between events, equipment failures
- **Poisson**: Count data, rare events
- **Binomial**: Success/failure outcomes
- **Triangular**: Known min/max/mode values

### Why This Matters for VALORANT
Understanding these distributions helps:
- **Identify outliers**: Players performing exceptionally well/poorly
- **Predict performance**: What ranges to expect from typical players
- **Balance game design**: Ensure mechanics reward balanced play
- **Recruitment**: Understand what "normal" pro-level performance looks like

---

## Dataset Structure

The bronze dataset contains CSV files organized as:
```
bronze/
├── event_id=1/
│   ├── region=na/
│   │   ├── map=haven/
│   │   │   ├── agent=jett/
│   │   │   │   └── snapshot_date=2026-02-28/
│   │   │   │       └── data.csv
```

Each `data.csv` contains player statistics including:
- `player_id`: Unique player identifier
- `average_combat_score`: ACS for that match

---

## Files in This Repository

- **VLR.py**: Main script for real data analysis
- **VLR_distributions.py**: Synthetic distribution demonstration
- **.gitignore**: Excludes the large bronze/ dataset folder
- **README.md**: This file

---

## Requirements

```
pandas
matplotlib
seaborn
numpy
```

Install with:
```bash
pip install pandas matplotlib seaborn numpy
```

---

## Notes

- The bronze folder is ignored by git due to its large size (contains thousands of CSV files)
- VLR.py deduplicates by player_id to avoid counting the same player multiple times
- VLR_distributions.py uses seed=42 for reproducible results
- All visualizations include mean and median reference lines for better interpretation

---

## Author Notes

These scripts demonstrate how statistical analysis can provide insights into competitive gaming performance and highlight the importance of understanding data distributions when working with large datasets.
