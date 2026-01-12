import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnergyConsumptionAnalyzer:
    """
    Comprehensive Energy Consumption & Load Forecast Analyzer
    for Smart Grid / Power Systems analysis
    """
    
    def __init__(self, data=None):
        """
        Initialize the analyzer with data or generate synthetic data
        """
        self.data = data
        self.analysis_results = {}
        self.efficiency_scores = {}
        
    def generate_synthetic_data(self, n_days=30, seed=42):
        """
        Generate synthetic hourly energy consumption data for analysis
        """
        np.random.seed(seed)
        
        # Generate date range
        start_date = datetime.now() - timedelta(days=n_days)
        dates = pd.date_range(start=start_date, periods=n_days*24, freq='H')
        
        # Base consumption patterns
        base_load = 500  # kW base load
        
        # Time-based patterns
        hours = dates.hour
        day_of_week = dates.dayofweek
        
        # Create consumption patterns
        consumption = []
        
        for i, (hour, weekday) in enumerate(zip(hours, day_of_week)):
            # Base pattern
            if 0 <= hour < 6:  # Night (low consumption)
                load = base_load * np.random.uniform(0.3, 0.5)
            elif 6 <= hour < 9:  # Morning peak
                load = base_load * np.random.uniform(1.2, 1.5)
            elif 9 <= hour < 17:  # Daytime
                load = base_load * np.random.uniform(0.8, 1.2)
            elif 17 <= hour < 22:  # Evening peak
                load = base_load * np.random.uniform(1.5, 2.0)
            else:  # Late evening
                load = base_load * np.random.uniform(0.6, 0.9)
            
            # Weekday vs weekend adjustment
            if weekday >= 5:  # Weekend
                load *= np.random.uniform(0.7, 0.9)
            
            # Add some anomalies (5% of data)
            if np.random.random() < 0.05:
                load *= np.random.uniform(1.5, 3.0)  # Spike
            elif np.random.random() < 0.03:
                load *= np.random.uniform(0.1, 0.5)  # Drop
            
            # Add noise
            load += np.random.normal(0, base_load * 0.1)
            
            consumption.append(max(load, 50))  # Ensure positive values
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'timestamp': dates,
            'consumption_kw': consumption,
            'hour': hours,
            'day_of_week': day_of_week,
            'is_weekday': day_of_week < 5,
            'date': dates.date
        })
        
        print(f"Generated {len(self.data)} hourly records")
        return self.data
    
    def peak_off_peak_analysis(self):
        """
        Analyze peak vs off-peak load patterns
        """
        df = self.data.copy()
        
        # Define peak hours (typically 8-20 on weekdays)
        df['is_peak_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 20) & 
                             (df['is_weekday'] == True))
        
        # Calculate statistics
        peak_stats = df[df['is_peak_hour']]['consumption_kw'].describe()
        off_peak_stats = df[~df['is_peak_hour']]['consumption_kw'].describe()
        
        # Peak to off-peak ratio
        peak_to_offpeak = peak_stats['mean'] / off_peak_stats['mean']
        
        self.analysis_results['peak_analysis'] = {
            'peak_hours_stats': peak_stats,
            'off_peak_hours_stats': off_peak_stats,
            'peak_to_offpeak_ratio': peak_to_offpeak,
            'peak_hour_threshold': 0.8  # 80th percentile as peak threshold
        }
        
        return self.analysis_results['peak_analysis']
    
    def rolling_average_analysis(self, window_sizes=[24, 168]):
        """
        Calculate rolling average load trends
        window_sizes: list of window sizes in hours (24h, 168h=1 week)
        """
        df = self.data.copy()
        
        rolling_stats = {}
        for window in window_sizes:
            df[f'rolling_{window}h'] = df['consumption_kw'].rolling(
                window=window, center=True).mean()
            df[f'rolling_{window}h_std'] = df['consumption_kw'].rolling(
                window=window, center=True).std()
            
            rolling_stats[f'rolling_{window}h'] = {
                'mean': df[f'rolling_{window}h'].mean(),
                'std': df[f'rolling_{window}h'].std(),
                'trend': self._calculate_trend(df[f'rolling_{window}h'].dropna())
            }
        
        self.data = df
        self.analysis_results['rolling_analysis'] = rolling_stats
        return rolling_stats
    
    def _calculate_trend(self, series):
        """Calculate linear trend of a series"""
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    def weekday_weekend_comparison(self):
        """
        Compare consumption patterns between weekdays and weekends
        """
        df = self.data.copy()
        
        # Group by hour and weekday/weekend
        comparison = df.groupby(['hour', 'is_weekday'])['consumption_kw'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Calculate differences
        weekday_avg = df[df['is_weekday']]['consumption_kw'].mean()
        weekend_avg = df[~df['is_weekday']]['consumption_kw'].mean()
        reduction_percentage = ((weekday_avg - weekend_avg) / weekday_avg) * 100
        
        self.analysis_results['weekday_weekend_comparison'] = {
            'comparison_table': comparison,
            'weekday_average': weekday_avg,
            'weekend_average': weekend_avg,
            'reduction_percentage': reduction_percentage,
            'peak_hour_difference': self._calculate_peak_hour_difference(df)
        }
        
        return self.analysis_results['weekday_weekend_comparison']
    
    def _calculate_peak_hour_difference(self, df):
        """Calculate peak hour differences between weekdays and weekends"""
        weekday_peak_hour = df[df['is_weekday']].groupby('hour')['consumption_kw'].mean().idxmax()
        weekend_peak_hour = df[~df['is_weekday']].groupby('hour')['consumption_kw'].mean().idxmax()
        return {
            'weekday_peak_hour': weekday_peak_hour,
            'weekend_peak_hour': weekend_peak_hour,
            'hour_difference': abs(weekday_peak_hour - weekend_peak_hour)
        }
    
    def anomaly_detection(self, threshold_std=3):
        """
        Detect anomalies using statistical thresholds (Z-score method)
        """
        df = self.data.copy()
        
        # Calculate Z-scores
        df['z_score'] = np.abs(stats.zscore(df['consumption_kw']))
        
        # Identify anomalies
        anomalies = df[df['z_score'] > threshold_std].copy()
        anomalies['anomaly_type'] = anomalies.apply(
            lambda row: 'High' if row['consumption_kw'] > df['consumption_kw'].mean() 
            else 'Low', axis=1
        )
        
        # Calculate anomaly statistics
        anomaly_stats = {
            'total_anomalies': len(anomalies),
            'high_anomalies': len(anomalies[anomalies['anomaly_type'] == 'High']),
            'low_anomalies': len(anomalies[anomalies['anomaly_type'] == 'Low']),
            'anomaly_percentage': (len(anomalies) / len(df)) * 100,
            'anomalies': anomalies[['timestamp', 'consumption_kw', 'z_score', 'anomaly_type']]
        }
        
        self.analysis_results['anomaly_detection'] = anomaly_stats
        return anomaly_stats
    
    def energy_efficiency_scoring(self):
        """
        Calculate energy efficiency scores based on multiple factors
        """
        df = self.data.copy()
        
        # Calculate various efficiency metrics
        scores = {}
        
        # 1. Load Factor Score (ratio of average to peak load)
        peak_load = df['consumption_kw'].max()
        avg_load = df['consumption_kw'].mean()
        load_factor = avg_load / peak_load
        scores['load_factor_score'] = min(load_factor * 100, 100)
        
        # 2. Peak Reduction Score
        peak_hours = df[(df['hour'] >= 8) & (df['hour'] <= 20)]
        off_peak_hours = df[(df['hour'] < 8) | (df['hour'] > 20)]
        peak_ratio = peak_hours['consumption_kw'].mean() / off_peak_hours['consumption_kw'].mean()
        scores['peak_reduction_score'] = max(0, 100 - (peak_ratio - 1) * 50)
        
        # 3. Weekend Reduction Score
        weekday_avg = df[df['is_weekday']]['consumption_kw'].mean()
        weekend_avg = df[~df['is_weekday']]['consumption_kw'].mean()
        weekend_reduction = ((weekday_avg - weekend_avg) / weekday_avg) * 100
        scores['weekend_reduction_score'] = min(max(weekend_reduction, 0), 100)
        
        # 4. Consistency Score (inverse of coefficient of variation)
        cv = df['consumption_kw'].std() / df['consumption_kw'].mean()
        scores['consistency_score'] = max(0, 100 - cv * 50)
        
        # Calculate overall efficiency score (weighted average)
        weights = {
            'load_factor_score': 0.3,
            'peak_reduction_score': 0.3,
            'weekend_reduction_score': 0.2,
            'consistency_score': 0.2
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores.keys())
        scores['overall_efficiency_score'] = overall_score
        
        # Assign efficiency grade
        if overall_score >= 80:
            grade = 'A (Excellent)'
        elif overall_score >= 70:
            grade = 'B (Good)'
        elif overall_score >= 60:
            grade = 'C (Average)'
        elif overall_score >= 50:
            grade = 'D (Poor)'
        else:
            grade = 'F (Very Poor)'
        
        scores['efficiency_grade'] = grade
        
        self.efficiency_scores = scores
        self.analysis_results['efficiency_scores'] = scores
        
        return scores
    
    def detect_power_wastage(self):
        """
        Detect potential power wastage patterns
        """
        df = self.data.copy()
        
        wastage_patterns = []
        
        # Pattern 1: High off-peak consumption
        off_peak_hours = df[(df['hour'] < 6) | (df['hour'] > 22)]
        off_peak_avg = off_peak_hours['consumption_kw'].mean()
        overall_avg = df['consumption_kw'].mean()
        
        if off_peak_avg > overall_avg * 0.7:
            wastage_patterns.append({
                'type': 'High off-peak consumption',
                'severity': 'Medium',
                'description': f'Off-peak consumption ({off_peak_avg:.1f} kW) is high relative to overall average',
                'suggestion': 'Implement better scheduling of non-essential equipment'
            })
        
        # Pattern 2: Weekend consumption similar to weekdays
        weekday_avg = df[df['is_weekday']]['consumption_kw'].mean()
        weekend_avg = df[~df['is_weekday']]['consumption_kw'].mean()
        
        if weekend_avg > weekday_avg * 0.8:
            wastage_patterns.append({
                'type': 'High weekend consumption',
                'severity': 'Low',
                'description': f'Weekend consumption is {weekend_avg/weekday_avg*100:.1f}% of weekday consumption',
                'suggestion': 'Review weekend operation schedules'
            })
        
        # Pattern 3: Frequent high peaks
        peak_threshold = df['consumption_kw'].quantile(0.9)
        peak_count = len(df[df['consumption_kw'] > peak_threshold])
        peak_percentage = (peak_count / len(df)) * 100
        
        if peak_percentage > 15:
            wastage_patterns.append({
                'type': 'Frequent peak loads',
                'severity': 'High',
                'description': f'{peak_percentage:.1f}% of hours exceed peak threshold',
                'suggestion': 'Implement load shedding or peak shaving strategies'
            })
        
        self.analysis_results['wastage_patterns'] = wastage_patterns
        return wastage_patterns
    
    def identify_peak_load_risks(self):
        """
        Identify peak load risks and potential issues
        """
        df = self.data.copy()
        
        risks = []
        
        # Risk 1: Maximum capacity proximity
        max_consumption = df['consumption_kw'].max()
        capacity = max_consumption * 1.5  # Assuming 50% safety margin
        capacity_utilization = (max_consumption / capacity) * 100
        
        if capacity_utilization > 80:
            risks.append({
                'type': 'Capacity Risk',
                'severity': 'High',
                'description': f'Peak load reaches {capacity_utilization:.1f}% of assumed capacity',
                'recommendation': 'Consider capacity upgrade or load management'
            })
        
        # Risk 2: Rapid load changes
        df['hourly_change'] = df['consumption_kw'].diff().abs()
        rapid_changes = df[df['hourly_change'] > df['consumption_kw'].mean() * 0.3]
        
        if len(rapid_changes) > 5:
            risks.append({
                'type': 'Volatility Risk',
                'severity': 'Medium',
                'description': f'{len(rapid_changes)} instances of rapid load changes (>30% change)',
                'recommendation': 'Implement smoother load transition controls'
            })
        
        # Risk 3: Consecutive peak hours
        df['is_peak'] = df['consumption_kw'] > df['consumption_kw'].quantile(0.8)
        peak_groups = df['is_peak'].astype(int).groupby(
            (df['is_peak'] != df['is_peak'].shift()).cumsum()
        ).transform('sum')
        max_consecutive_peaks = peak_groups.max()
        
        if max_consecutive_peaks >= 4:
            risks.append({
                'type': 'Sustained Peak Risk',
                'severity': 'Medium',
                'description': f'Peak loads sustained for {max_consecutive_peaks} consecutive hours',
                'recommendation': 'Implement demand response strategies'
            })
        
        self.analysis_results['peak_load_risks'] = risks
        return risks
    
    def suggest_demand_balancing_strategies(self):
        """
        Suggest demand balancing strategies based on analysis
        """
        strategies = []
        
        # Analyze patterns to suggest strategies
        peak_analysis = self.analysis_results.get('peak_analysis', {})
        efficiency = self.efficiency_scores
        
        if peak_analysis.get('peak_to_offpeak_ratio', 0) > 1.5:
            strategies.append({
                'strategy': 'Time-of-Use Pricing',
                'description': 'Implement differential pricing to shift load to off-peak hours',
                'expected_impact': '20-30% peak reduction',
                'implementation': 'Medium'
            })
        
        if efficiency.get('weekend_reduction_score', 0) < 20:
            strategies.append({
                'strategy': 'Weekend Load Scheduling',
                'description': 'Schedule non-essential operations to weekdays',
                'expected_impact': '15-25% weekend load reduction',
                'implementation': 'Easy'
            })
        
        if efficiency.get('consistency_score', 0) < 60:
            strategies.append({
                'strategy': 'Load Leveling',
                'description': 'Use energy storage systems to smooth load profile',
                'expected_impact': 'Improved consistency, reduced peaks',
                'implementation': 'Hard (requires investment)'
            })
        
        # Always suggest these basic strategies
        strategies.extend([
            {
                'strategy': 'Energy Efficiency Audit',
                'description': 'Comprehensive audit to identify wastage points',
                'expected_impact': '10-20% overall reduction',
                'implementation': 'Easy'
            },
            {
                'strategy': 'Automated Demand Response',
                'description': 'Automated systems to reduce load during peaks',
                'expected_impact': '15-25% peak reduction',
                'implementation': 'Medium'
            }
        ])
        
        self.analysis_results['balancing_strategies'] = strategies
        return strategies
    
    def generate_report(self):
        """
        Generate comprehensive analysis report
        """
        print("=" * 70)
        print("ENERGY CONSUMPTION & LOAD FORECAST ANALYSIS REPORT")
        print("=" * 70)
        
        # Basic Statistics
        print("\n1. BASIC STATISTICS")
        print("-" * 40)
        print(f"Total hours analyzed: {len(self.data)}")
        print(f"Total energy consumed: {self.data['consumption_kw'].sum():.0f} kWh")
        print(f"Average consumption: {self.data['consumption_kw'].mean():.1f} kW")
        print(f"Peak consumption: {self.data['consumption_kw'].max():.1f} kW")
        print(f"Minimum consumption: {self.data['consumption_kw'].min():.1f} kW")
        
        # Peak Analysis
        if 'peak_analysis' in self.analysis_results:
            print("\n2. PEAK VS OFF-PEAK ANALYSIS")
            print("-" * 40)
            pa = self.analysis_results['peak_analysis']
            print(f"Peak to Off-peak ratio: {pa['peak_to_offpeak_ratio']:.2f}")
            print(f"Peak hours average: {pa['peak_hours_stats']['mean']:.1f} kW")
            print(f"Off-peak hours average: {pa['off_peak_hours_stats']['mean']:.1f} kW")
        
        # Efficiency Scores
        if self.efficiency_scores:
            print("\n3. ENERGY EFFICIENCY SCORING")
            print("-" * 40)
            for key, value in self.efficiency_scores.items():
                if 'score' in key and not key.startswith('overall'):
                    print(f"{key.replace('_', ' ').title()}: {value:.1f}")
            print(f"\nOverall Efficiency Score: {self.efficiency_scores.get('overall_efficiency_score', 0):.1f}")
            print(f"Efficiency Grade: {self.efficiency_scores.get('efficiency_grade', 'N/A')}")
        
        # Wastage Patterns
        if 'wastage_patterns' in self.analysis_results:
            print("\n4. POWER WASTAGE PATTERNS DETECTED")
            print("-" * 40)
            for pattern in self.analysis_results['wastage_patterns']:
                print(f"• {pattern['type']} ({pattern['severity']}): {pattern['description']}")
        
        # Peak Load Risks
        if 'peak_load_risks' in self.analysis_results:
            print("\n5. PEAK LOAD RISKS IDENTIFIED")
            print("-" * 40)
            for risk in self.analysis_results['peak_load_risks']:
                print(f"• {risk['type']} ({risk['severity']}): {risk['description']}")
        
        # Recommendations
        if 'balancing_strategies' in self.analysis_results:
            print("\n6. RECOMMENDED DEMAND BALANCING STRATEGIES")
            print("-" * 40)
            for i, strategy in enumerate(self.analysis_results['balancing_strategies'], 1):
                print(f"{i}. {strategy['strategy']}")
                print(f"   Description: {strategy['description']}")
                print(f"   Expected Impact: {strategy['expected_impact']}")
                print(f"   Implementation: {strategy['implementation']}")
                print()
        
        print("=" * 70)
        print("END OF REPORT")
        print("=" * 70)
    
    def visualize_analysis(self):
        """
        Create comprehensive visualizations of the analysis
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Energy Consumption Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series with rolling averages
        ax1 = axes[0, 0]
        ax1.plot(self.data['timestamp'], self.data['consumption_kw'], 
                alpha=0.5, label='Hourly Consumption', linewidth=0.5)
        if 'rolling_24h' in self.data.columns:
            ax1.plot(self.data['timestamp'], self.data['rolling_24h'], 
                    'r-', label='24h Rolling Average', linewidth=2)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Consumption (kW)')
        ax1.set_title('Hourly Consumption with Trends')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Daily profile
        ax2 = axes[0, 1]
        daily_profile = self.data.groupby('hour')['consumption_kw'].mean()
        ax2.plot(daily_profile.index, daily_profile.values, 'b-', linewidth=2)
        ax2.fill_between(daily_profile.index, 0, daily_profile.values, alpha=0.3)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Consumption (kW)')
        ax2.set_title('Average Daily Load Profile')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 3))
        
        # Plot 3: Weekday vs Weekend comparison
        ax3 = axes[1, 0]
        weekday_profile = self.data[self.data['is_weekday']].groupby('hour')['consumption_kw'].mean()
        weekend_profile = self.data[~self.data['is_weekday']].groupby('hour')['consumption_kw'].mean()
        ax3.plot(weekday_profile.index, weekday_profile.values, 'g-', 
                label='Weekdays', linewidth=2)
        ax3.plot(weekend_profile.index, weekend_profile.values, 'r-', 
                label='Weekends', linewidth=2)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Consumption (kW)')
        ax3.set_title('Weekday vs Weekend Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 3))
        
        # Plot 4: Anomaly detection
        ax4 = axes[1, 1]
        ax4.plot(self.data['timestamp'], self.data['consumption_kw'], 
                'b-', alpha=0.5, linewidth=0.5)
        
        if 'anomaly_detection' in self.analysis_results:
            anomalies = self.analysis_results['anomaly_detection']['anomalies']
            high_anomalies = anomalies[anomalies['anomaly_type'] == 'High']
            low_anomalies = anomalies[anomalies['anomaly_type'] == 'Low']
            
            ax4.scatter(high_anomalies['timestamp'], high_anomalies['consumption_kw'], 
                       color='red', s=50, label='High Anomalies', zorder=5)
            ax4.scatter(low_anomalies['timestamp'], low_anomalies['consumption_kw'], 
                       color='orange', s=50, label='Low Anomalies', zorder=5)
        
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Consumption (kW)')
        ax4.set_title('Anomaly Detection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Efficiency scores (bar chart)
        ax5 = axes[2, 0]
        if self.efficiency_scores:
            score_labels = [k.replace('_score', '').replace('_', ' ').title() 
                          for k in self.efficiency_scores.keys() 
                          if 'score' in k and not k.startswith('overall')]
            scores = [v for k, v in self.efficiency_scores.items() 
                     if 'score' in k and not k.startswith('overall')]
            
            bars = ax5.bar(score_labels, scores, color=['green', 'blue', 'orange', 'purple'])
            ax5.set_ylabel('Score')
            ax5.set_title('Energy Efficiency Scores')
            ax5.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Peak load distribution
        ax6 = axes[2, 1]
        peak_threshold = self.data['consumption_kw'].quantile(0.8)
        ax6.hist(self.data['consumption_kw'], bins=30, alpha=0.7, edgecolor='black')
        ax6.axvline(peak_threshold, color='red', linestyle='--', 
                   label=f'Peak Threshold ({peak_threshold:.0f} kW)')
        ax6.set_xlabel('Consumption (kW)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Load Distribution & Peak Threshold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage and demonstration
def main():
    """
    Main function to demonstrate the Energy Consumption Analyzer
    """
    print("Initializing Energy Consumption & Load Forecast Analyzer...\n")
    
    # Create analyzer instance
    analyzer = EnergyConsumptionAnalyzer()
    
    # Generate synthetic data (30 days of hourly data)
    print("Generating synthetic energy consumption data...")
    data = analyzer.generate_synthetic_data(n_days=30)
    
    # Perform all analyses
    print("\nPerforming comprehensive analysis...")
    print("-" * 50)
    
    print("1. Analyzing peak vs off-peak patterns...")
    analyzer.peak_off_peak_analysis()
    
    print("2. Calculating rolling averages...")
    analyzer.rolling_average_analysis()
    
    print("3. Comparing weekday vs weekend consumption...")
    analyzer.weekday_weekend_comparison()
    
    print("4. Detecting anomalies...")
    analyzer.anomaly_detection()
    
    print("5. Calculating energy efficiency scores...")
    analyzer.energy_efficiency_scoring()
    
    print("6. Detecting power wastage patterns...")
    analyzer.detect_power_wastage()
    
    print("7. Identifying peak load risks...")
    analyzer.identify_peak_load_risks()
    
    print("8. Suggesting demand balancing strategies...")
    analyzer.suggest_demand_balancing_strategies()
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 70)
    analyzer.generate_report()
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.visualize_analysis()
    
    # Export capabilities
    print("\nAdditional Features:")
    print("-" * 40)
    print("• Data can be exported to CSV for further analysis")
    print("• All analysis results are stored in analyzer.analysis_results")
    print("• Custom thresholds can be adjusted in each method")
    print("• Real data can be loaded using pandas.read_csv()")
    
    return analyzer

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = main()
    
    # Example of how to use with real data
    print("\n" + "=" * 70)
    print("USING WITH REAL DATA - EXAMPLE:")
    print("=" * 70)
    print("""
    # Load your own data:
    # analyzer = EnergyConsumptionAnalyzer()
    # real_data = pd.read_csv('your_energy_data.csv')
    # real_data['timestamp'] = pd.to_datetime(real_data['timestamp'])
    # analyzer.data = real_data
    # analyzer.run_all_analyses()
    """)

# Save data and results example
def export_results(analyzer, filename_prefix="energy_analysis"):
    """
    Export analysis results to files
    """
    # Save data to CSV
    analyzer.data.to_csv(f"{filename_prefix}_data.csv", index=False)
    
    # Save analysis results summary
    summary = {
        'efficiency_scores': analyzer.efficiency_scores,
        'peak_analysis': analyzer.analysis_results.get('peak_analysis', {}),
        'anomaly_summary': analyzer.analysis_results.get('anomaly_detection', {}),
        'wastage_patterns': analyzer.analysis_results.get('wastage_patterns', []),
        'peak_load_risks': analyzer.analysis_results.get('peak_load_risks', []),
        'strategies': analyzer.analysis_results.get('balancing_strategies', [])
    }
    
    import json
    with open(f"{filename_prefix}_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults exported to {filename_prefix}_data.csv and {filename_prefix}_summary.json")