# bayesian_journey.py
"""
Interactive Bayesian Prior-to-Posterior Journey for Google Colab
Helps users understand how priors and data combine to form posteriors
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Color scheme matching your viridish palette
COLORS = {
    'prior': '#481C6E',      # Purple
    'likelihood': '#25AB81',  # Teal  
    'posterior': '#B0DD2F',   # Yellow-green
    'observed': '#FF6B6B',    # Coral red
    'fill_prior': 'rgba(72, 28, 110, 0.2)',
    'fill_posterior': 'rgba(176, 221, 47, 0.2)'
}

class BayesianJourney:
    """Interactive widget for understanding Bayesian updating"""
    
    def __init__(self):
        self.prior_alpha = 1.0
        self.prior_beta = 1.0
        self.successes = 0
        self.failures = 0
        self.history = []  # Track the journey
        self.max_history = 5000 #50  # Limit history length
        
        # Create all widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create all interactive widgets"""
        
        # === PRIOR DESIGN SECTION ===
        self.prior_header = widgets.HTML(
            value="<h3>📊 Step 1: Design Your Prior Belief</h3>" +
                  "<p style='color: gray;'>What do you believe about the conversion rate before seeing any data?  Enter your expected number of successes+1 (alpha) and failures+1 (beta).  Your inputs define the shape of a prior distribution that is centered on your expected outcome value, and that has a wider shape the more uncertain you are.  </p>"
        )
        
        # Prior preset buttons
        self.preset_label = widgets.HTML(value="<b>Quick Presets:</b>")
        self.preset_uniform = widgets.Button(
            description="🤷 Uniform\n(No idea)",
            layout=widgets.Layout(width='150px', height='60px'),
            button_style='info',
            tooltip="Beta(1,1) - All conversion rates equally likely"
        )
        self.preset_pessimistic = widgets.Button(
            description="😟 Pessimistic\n(Expect low)",
            layout=widgets.Layout(width='150px', height='60px'),
            button_style='warning',
            tooltip="Beta(20,200) - Expect around 10% conversion"
        )
        self.preset_optimistic = widgets.Button(
            description="😊 Optimistic\n(Expect high)",
            layout=widgets.Layout(width='150px', height='60px'),
            button_style='success',
            tooltip="Beta(200,100) - Expect around 67% conversion"
        )
        self.preset_confident = widgets.Button(
            description="📈 Industry CR \n(~15% typical)",
            layout=widgets.Layout(width='150px', height='60px'),
            button_style='primary',
            tooltip="Beta(150,850) - Industry standard ~15%"
        )
        
        # Manual prior controls
        self.alpha_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=5000.0,
            step=0.1,
            description='α (alpha):',
            continuous_update=True,
            readout_format='.1f',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.beta_slider = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=5000.0,
            step=0.1,
            description='β (beta):',
            continuous_update=True,
            readout_format='.1f',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        # Prior interpretation
        self.prior_interpretation = widgets.HTML(value=self._get_prior_interpretation())
        
        # === DATA COLLECTION SECTION ===
        self.data_header = widgets.HTML(
            value="<h3>🎯 Step 2: Observe Data & Update Beliefs</h3>" +
                  "<p style='color: gray;'>Add observations one at a time and watch your beliefs update!  </p>"
        )
        
        # Data entry buttons
        self.add_success = widgets.Button(
            description="✅ Add Success",
            button_style='success',
            icon='plus-circle',
            layout=widgets.Layout(width='150px', height='40px'),
            tooltip="Customer converted!"
        )
        
        self.add_failure = widgets.Button(
            description="❌ Add Failure", 
            button_style='danger',
            icon='minus-circle',
            layout=widgets.Layout(width='150px', height='40px'),
            tooltip="Customer didn't convert"
        )
        
        # Batch add section
        self.batch_successes = widgets.IntText(
            value=0,
            description='Successes:',
            layout=widgets.Layout(width='150px')
        )
        self.batch_failures = widgets.IntText(
            value=0,
            description='Failures:',
            layout=widgets.Layout(width='150px')
        )
        self.add_batch = widgets.Button(
            description="Add Batch",
            button_style='primary',
            icon='database',
            layout=widgets.Layout(width='120px')
        )
        
        # Reset button
        self.reset_data = widgets.Button(
            description="🔄 Reset Data",
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(width='150px', height='40px'),
            tooltip="Clear all data and start over"
        )
        
        # Data counter display
        self.data_display = widgets.HTML(value=self._get_data_display())
        
        # === VISUALIZATION SECTION ===
        self.viz_header = widgets.HTML(
            value="<h3>📈 Visualization</h3>"
        )
        
        # Visualization options
        self.show_prior = widgets.Checkbox(value=True, description="Show Prior", indent=False)
        self.show_likelihood = widgets.Checkbox(value=False, description="Show Likelihood", indent=False)
        self.show_credible = widgets.Checkbox(value=True, description="Show 95% Credible Interval", indent=False)
        self.show_map = widgets.Checkbox(value=True, description="Show MAP Estimate", indent=False)
        
        # Output areas
        self.plot_output = widgets.Output(layout=widgets.Layout(width='100%', height='500px'))
        self.stats_output = widgets.HTML(value=self._get_stats_display())
        self.history_output = widgets.HTML(value=self._get_history_display())
        
        # === EVENT HANDLERS ===
        self.preset_uniform.on_click(lambda b: self._set_prior(1, 1))
        self.preset_pessimistic.on_click(lambda b: self._set_prior(20, 200))
        self.preset_optimistic.on_click(lambda b: self._set_prior(200, 100))
        self.preset_confident.on_click(lambda b: self._set_prior(150, 850))
        
        self.alpha_slider.observe(self._on_prior_change, 'value')
        self.beta_slider.observe(self._on_prior_change, 'value')
        
        self.add_success.on_click(lambda b: self._add_observation(1, 0))
        self.add_failure.on_click(lambda b: self._add_observation(0, 1))
        self.add_batch.on_click(self._add_batch_observation)
        self.reset_data.on_click(self._reset_data)
        
        self.show_prior.observe(self._update_plot, 'value')
        self.show_likelihood.observe(self._update_plot, 'value')
        self.show_credible.observe(self._update_plot, 'value')
        self.show_map.observe(self._update_plot, 'value')
        
    def _set_prior(self, alpha, beta):
        """Set prior parameters from preset"""
        self.alpha_slider.value = alpha
        self.beta_slider.value = beta
        
    def _on_prior_change(self, change):
        """Handle prior parameter changes"""
        self.prior_alpha = self.alpha_slider.value
        self.prior_beta = self.beta_slider.value
        self.prior_interpretation.value = self._get_prior_interpretation()
        self._update_all()
        
    def _add_observation(self, success, failure):
        """Add a single observation"""
        self.successes += success
        self.failures += failure
        
        # Add to history
        if success:
            self.history.append({'type': 'success', 'total_s': self.successes, 'total_f': self.failures})
        else:
            self.history.append({'type': 'failure', 'total_s': self.successes, 'total_f': self.failures})
            
        # Limit history length
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
        self._update_all()
        
    def _add_batch_observation(self, b):
        """Add batch observations"""
        s = self.batch_successes.value
        f = self.batch_failures.value
        
        if s > 0 or f > 0:
            self.successes += s
            self.failures += f
            
            self.history.append({
                'type': 'batch',
                'batch_s': s,
                'batch_f': f,
                'total_s': self.successes,
                'total_f': self.failures
            })
            
            # Reset batch inputs
            self.batch_successes.value = 0
            self.batch_failures.value = 0
            
            self._update_all()
            
    def _reset_data(self, b):
        """Reset all data"""
        self.successes = 0
        self.failures = 0
        self.history = []
        self._update_all()
        
    def _update_all(self):
        """Update all displays"""
        self.data_display.value = self._get_data_display()
        self.stats_output.value = self._get_stats_display()
        self.history_output.value = self._get_history_display()
        self._update_plot()
        
    def _update_plot(self, change=None):
        """Update the main visualization"""
        with self.plot_output:
            clear_output(wait=True)
            
            # Create figure
            fig = go.Figure()
            
            # X-axis values
            x = np.linspace(0, 1, 500)
            
            # Prior distribution
            prior_y = stats.beta.pdf(x, self.prior_alpha, self.prior_beta)
            
            # Posterior distribution
            post_alpha = self.prior_alpha + self.successes
            post_beta = self.prior_beta + self.failures
            posterior_y = stats.beta.pdf(x, post_alpha, post_beta)
            
            # Add posterior first (so it's behind)
            if self.successes > 0 or self.failures > 0:
                fig.add_trace(go.Scatter(
                    x=x, y=posterior_y,
                    name='Posterior',
                    line=dict(color=COLORS['posterior'], width=3),
                    fill='tozeroy',
                    fillcolor=COLORS['fill_posterior']
                ))
                
                # Add credible interval if requested
                if self.show_credible.value:
                    ci_lower = stats.beta.ppf(0.025, post_alpha, post_beta)
                    ci_upper = stats.beta.ppf(0.975, post_alpha, post_beta)
                    
                    # Shade the credible interval
                    ci_mask = (x >= ci_lower) & (x <= ci_upper)
                    fig.add_trace(go.Scatter(
                        x=x[ci_mask],
                        y=posterior_y[ci_mask],
                        fill='tozeroy',
                        fillcolor='rgba(176, 221, 47, 0.4)',
                        line=dict(width=0),
                        showlegend=True,
                        name=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]'
                    ))
                
                # Add MAP estimate
                if self.show_map.value and post_alpha > 1 and post_beta > 1:
                    map_estimate = (post_alpha - 1) / (post_alpha + post_beta - 2)
                    map_y = stats.beta.pdf(map_estimate, post_alpha, post_beta)
                    fig.add_trace(go.Scatter(
                        x=[map_estimate, map_estimate],
                        y=[0, map_y],
                        mode='lines',
                        line=dict(color=COLORS['posterior'], width=2, dash='dash'),
                        name=f'MAP: {map_estimate:.3f}'
                    ))
            
            # Add prior if requested
            if self.show_prior.value:
                fig.add_trace(go.Scatter(
                    x=x, y=prior_y,
                    name='Prior',
                    line=dict(color=COLORS['prior'], width=2, dash='dot'),
                    fill='tozeroy',
                    fillcolor=COLORS['fill_prior']
                ))
            
            # Add likelihood if requested and we have data
            if self.show_likelihood.value and (self.successes > 0 or self.failures > 0):
                # Likelihood is proportional to x^successes * (1-x)^failures
                likelihood_y = (x ** self.successes) * ((1 - x) ** self.failures)
                # Normalize for visualization
                if likelihood_y.max() > 0:
                    likelihood_y = likelihood_y / likelihood_y.max() * posterior_y.max() * 0.8
                    
                fig.add_trace(go.Scatter(
                    x=x, y=likelihood_y,
                    name=f'Likelihood (n={self.successes + self.failures})',
                    line=dict(color=COLORS['likelihood'], width=2, dash='dashdot')
                ))
            
            # Add observed proportion
            if self.successes + self.failures > 0:
                observed_prop = self.successes / (self.successes + self.failures)
                fig.add_trace(go.Scatter(
                    x=[observed_prop],
                    y=[0],
                    mode='markers',
                    marker=dict(size=12, color=COLORS['observed'], symbol='diamond'),
                    name=f'Observed: {observed_prop:.3f}'
                ))
            
            # Update layout
            title_text = "Bayesian Belief Update"
            if self.successes + self.failures > 0:
                title_text += f" | Data: {self.successes} successes, {self.failures} failures"
            
            fig.update_layout(
                title=title_text,
                xaxis_title="Conversion Rate (θ)",
                yaxis_title="Probability Density",
                template="plotly_white",
                height=400,
                hovermode='x unified',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            fig.show()
            
    def _get_prior_interpretation(self):
        """Get human-readable interpretation of prior"""
        mean = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        variance = (self.prior_alpha * self.prior_beta) / ((self.prior_alpha + self.prior_beta)**2 * (self.prior_alpha + self.prior_beta + 1))
        std = np.sqrt(variance)
        effective_n = self.prior_alpha + self.prior_beta
        
        interpretation = f"""
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0;'>
        <b>Prior Interpretation:</b><br>
        • <b>Expected rate:</b> {mean:.1%} ± {std:.1%}<br>
        • <b>Confidence:</b> Like having seen {effective_n:.0f} previous trials<br>
        • <b>In words:</b> {self._describe_prior()}
        </div>
        """
        return interpretation
    
    def _describe_prior(self):
        """Generate natural language description of prior"""
        mean = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        strength = self.prior_alpha + self.prior_beta
        
        if strength < 2.5:
            confidence = "very uncertain"
        elif strength < 10:
            confidence = "somewhat uncertain"
        elif strength < 50:
            confidence = "moderately confident"
        else:
            confidence = "very confident"
            
        if mean < 0.1:
            rate_desc = "very low"
        elif mean < 0.3:
            rate_desc = "low"
        elif mean < 0.7:
            rate_desc = "moderate"
        elif mean < 0.9:
            rate_desc = "high"
        else:
            rate_desc = "very high"
            
        return f"You're {confidence} the rate is {rate_desc} (~{mean:.0%})"
    
    def _get_data_display(self):
        """Get current data summary display"""
        total = self.successes + self.failures
        if total > 0:
            observed_rate = self.successes / total
            display_text = f"""
            <div style='background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h4 style='margin-top: 0;'>📊 Current Data:</h4>
            <div style='display: flex; justify-content: space-around;'>
                <div>✅ <b>Successes:</b> {self.successes}</div>
                <div>❌ <b>Failures:</b> {self.failures}</div>
                <div>📈 <b>Total:</b> {total}</div>
                <div>🎯 <b>Observed Rate:</b> {observed_rate:.1%}</div>
            </div>
            </div>
            """
        else:
            display_text = """
            <div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h4 style='margin-top: 0;'>📊 No Data Yet</h4>
            <p style='color: gray;'>Click the buttons above to add observations!</p>
            </div>
            """
        return display_text
    
    def _get_stats_display(self):
        """Get statistical summary display"""
        # Calculate posterior parameters
        post_alpha = self.prior_alpha + self.successes
        post_beta = self.prior_beta + self.failures
        
        # Prior stats
        prior_mean = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        
        # Posterior stats
        post_mean = post_alpha / (post_alpha + post_beta)
        post_var = (post_alpha * post_beta) / ((post_alpha + post_beta)**2 * (post_alpha + post_beta + 1))
        post_std = np.sqrt(post_var)
        
        if self.successes + self.failures > 0:
            # Calculate credible interval
            ci_lower = stats.beta.ppf(0.025, post_alpha, post_beta)
            ci_upper = stats.beta.ppf(0.975, post_alpha, post_beta)
            
            # Calculate information gain (KL divergence)
            kl_div = self._calculate_kl_divergence(
                self.prior_alpha, self.prior_beta,
                post_alpha, post_beta
            )
            
            stats_text = f"""
            <div style='background-color: #fff9e6; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h4 style='margin-top: 0;'>📈 Statistical Summary:</h4>
            <table style='width: 100%;'>
                <tr>
                    <td><b>Measure</b></td>
                    <td><b>Prior</b></td>
                    <td><b>Posterior</b></td>
                    <td><b>Change</b></td>
                </tr>
                <tr>
                    <td>Mean</td>
                    <td>{prior_mean:.3f}</td>
                    <td>{post_mean:.3f}</td>
                    <td>{post_mean - prior_mean:+.3f}</td>
                </tr>
                <tr>
                    <td>Std Dev</td>
                    <td>-</td>
                    <td>{post_std:.3f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>95% CI</td>
                    <td>-</td>
                    <td>[{ci_lower:.3f}, {ci_upper:.3f}]</td>
                    <td>Width: {ci_upper - ci_lower:.3f}</td>
                </tr>
            </table>
            <p style='margin-top: 10px;'><b>Information Gain:</b> {kl_div:.3f} bits</p>
            </div>
            """
        else:
            stats_text = """
            <div style='background-color: #fff9e6; padding: 15px; border-radius: 8px; margin: 10px 0;'>
            <h4 style='margin-top: 0;'>📈 Statistical Summary:</h4>
            <p style='color: gray;'>Add data to see how your beliefs update!</p>
            </div>
            """
            
        return stats_text
    
    def _get_history_display(self):
        """Get observation history display"""
        if len(self.history) == 0:
            return ""
            
        history_html = """
        <div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px; max-height: 150px; overflow-y: auto;'>
        <b>Recent History:</b><br>
        """
        
        for item in self.history[-5:]:  # Show last 5 items
            if item['type'] == 'success':
                history_html += f"✅ Success → Total: {item['total_s']}/{item['total_s'] + item['total_f']}<br>"
            elif item['type'] == 'failure':
                history_html += f"❌ Failure → Total: {item['total_s']}/{item['total_s'] + item['total_f']}<br>"
            else:  # batch
                history_html += f"📦 Batch: +{item['batch_s']}✅ +{item['batch_f']}❌ → Total: {item['total_s']}/{item['total_s'] + item['total_f']}<br>"
                
        history_html += "</div>"
        return history_html
    
    def _calculate_kl_divergence(self, alpha1, beta1, alpha2, beta2):
        """Calculate KL divergence between two Beta distributions"""
        from scipy.special import digamma, gammaln
        
        kl = gammaln(alpha2 + beta2) - gammaln(alpha1 + beta1)
        kl += gammaln(alpha1) - gammaln(alpha2)
        kl += gammaln(beta1) - gammaln(beta2)
        kl += (alpha2 - alpha1) * digamma(alpha2)
        kl += (beta2 - beta1) * digamma(beta2)
        kl -= (alpha2 + beta2 - alpha1 - beta1) * digamma(alpha2 + beta2)
        
        return max(0, kl) / np.log(2)  # Convert to bits
    
    def display(self):
        """Display the complete interface"""
        
        # Layout for prior section
        prior_section = widgets.VBox([
            self.prior_header,
            widgets.HBox([
                self.preset_label,
                self.preset_uniform,
                self.preset_pessimistic,
                self.preset_optimistic,
                self.preset_confident
            ]),
            widgets.VBox([
                self.alpha_slider,
                self.beta_slider
            ]),
            self.prior_interpretation
        ])
        
        # Layout for data section
        data_section = widgets.VBox([
            self.data_header,
            widgets.HBox([
                widgets.VBox([
                    widgets.HTML("<b>Single Observation:</b>"),
                    widgets.HBox([self.add_success, self.add_failure])
                ]),
                widgets.VBox([
                    widgets.HTML("<b>Batch Add:</b>"),
                    widgets.HBox([self.batch_successes, self.batch_failures, self.add_batch])
                ]),
                self.reset_data
            ]),
            self.data_display
        ])
        
        # Layout for visualization section
        viz_options = widgets.HBox([
            self.show_prior,
            self.show_likelihood,
            self.show_credible,
            self.show_map
        ])
        
        viz_section = widgets.VBox([
            self.viz_header,
            viz_options,
            self.plot_output,
            self.stats_output,
            self.history_output
        ])
        
        # Main container
        main_container = widgets.VBox([
            prior_section,
            widgets.HTML("<hr>"),
            data_section,
            widgets.HTML("<hr>"),
            viz_section
        ], layout=widgets.Layout(padding='20px'))
        
        # Initial plot
        self._update_plot()
        
        display(main_container)


def create_bayesian_journey():
    """Factory function to create and display the Bayesian journey"""
    journey = BayesianJourney()
    journey.display()
    return journey


# Additional comparison widget for A/B testing context
class BayesianABComparison:
    """Compare two variants with Bayesian updating"""
    
    def __init__(self):
        self.variant_a = {'alpha': 1, 'beta': 1, 'successes': 0, 'failures': 0}
        self.variant_b = {'alpha': 1, 'beta': 1, 'successes': 0, 'failures': 0}
        self.create_widgets()
        
    def create_widgets(self):
        """Create comparison interface"""
        
        self.header = widgets.HTML(
            value="<h3>⚖️ A/B Test Comparison</h3>" +
                  "<p style='color: gray;'>Add data to both variants and see the probability that B beats A</p>"
        )
        
        # Data input for A
        self.a_success = widgets.IntText(value=0, description='A Success:', layout=widgets.Layout(width='150px'))
        self.a_failure = widgets.IntText(value=0, description='A Failure:', layout=widgets.Layout(width='150px'))
        
        # Data input for B
        self.b_success = widgets.IntText(value=0, description='B Success:', layout=widgets.Layout(width='150px'))
        self.b_failure = widgets.IntText(value=0, description='B Failure:', layout=widgets.Layout(width='150px'))
        
        # Update button
        self.update_btn = widgets.Button(
            description='Update Comparison',
            button_style='primary',
            icon='refresh'
        )
        self.update_btn.on_click(self.update_comparison)
        
        # Output
        self.output = widgets.Output()
        
    def update_comparison(self, b):
        """Update the comparison plot"""
        with self.output:
            clear_output(wait=True)
            
            # Update data
            self.variant_a['successes'] = self.a_success.value
            self.variant_a['failures'] = self.a_failure.value
            self.variant_b['successes'] = self.b_success.value
            self.variant_b['failures'] = self.b_failure.value
            
            # Calculate posteriors
            a_alpha = self.variant_a['alpha'] + self.variant_a['successes']
            a_beta = self.variant_a['beta'] + self.variant_a['failures']
            b_alpha = self.variant_b['alpha'] + self.variant_b['successes']
            b_beta = self.variant_b['beta'] + self.variant_b['failures']
            
            # Monte Carlo simulation for P(B > A)
            n_samples = 10000
            samples_a = np.random.beta(a_alpha, a_beta, n_samples)
            samples_b = np.random.beta(b_alpha, b_beta, n_samples)
            prob_b_better = (samples_b > samples_a).mean()
            
            # Create plot
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Posterior Distributions', 'Difference (B - A)'))
            
            x = np.linspace(0, 1, 500)
            
            # Plot posteriors
            pdf_a = stats.beta.pdf(x, a_alpha, a_beta)
            pdf_b = stats.beta.pdf(x, b_alpha, b_beta)
            
            fig.add_trace(
                go.Scatter(x=x, y=pdf_a, name='Variant A', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=x, y=pdf_b, name='Variant B', line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # Plot difference distribution
            diff = samples_b - samples_a
            fig.add_trace(
                go.Histogram(x=diff, nbinsx=50, name='B - A', marker_color='purple', opacity=0.7),
                row=1, col=2
            )
            
            # Add zero line
            fig.add_vline(x=0, row=1, col=2, line_dash="dash", line_color="black")
            
            fig.update_layout(
                title=f"P(B > A) = {prob_b_better:.1%}",
                height=400,
                showlegend=True
            )
            
            fig.show()
            
            # Display stats
            display(HTML(f"""
            <div style='background-color: #e8f4f8; padding: 10px; border-radius: 5px;'>
            <b>Results:</b><br>
            • Variant A: {self.variant_a['successes']}/{self.variant_a['successes'] + self.variant_a['failures']} 
              = {a_alpha/(a_alpha+a_beta):.1%}<br>
            • Variant B: {self.variant_b['successes']}/{self.variant_b['successes'] + self.variant_b['failures']}
              = {b_alpha/(b_alpha+b_beta):.1%}<br>
            • <b>Probability B is better:</b> {prob_b_better:.1%}<br>
            • <b>Expected lift:</b> {(b_alpha/(b_alpha+b_beta) - a_alpha/(a_alpha+a_beta)):.1%}
            </div>
            """))
    
    def display(self):
        """Display the comparison interface"""
        container = widgets.VBox([
            self.header,
            widgets.HBox([
                widgets.VBox([self.a_success, self.a_failure]),
                widgets.VBox([self.b_success, self.b_failure]),
                self.update_btn
            ]),
            self.output
        ])
        display(container)
        self.update_comparison(None)
        

def create_ab_comparison():
    """Factory function for A/B comparison widget"""
    comparison = BayesianABComparison()
    comparison.display()
    return comparison
