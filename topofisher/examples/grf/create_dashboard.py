#!/usr/bin/env python3
"""
Create interactive HTML dashboard from TopoFisher results.

Usage:
    python create_dashboard.py
"""
import json
import numpy as np
from pathlib import Path


def compute_fisher_ellipse(F_00, F_11, F_01, n_sigma=1, n_points=100):
    """
    Compute confidence ellipse from Fisher matrix.

    Args:
        F_00, F_11, F_01: Fisher matrix elements
        n_sigma: Number of standard deviations (1, 2, or 3)
        n_points: Number of points for ellipse

    Returns:
        x, y: Arrays of ellipse coordinates (centered at origin)
    """
    # Fisher matrix
    F = np.array([[F_00, F_01], [F_01, F_11]])

    # Covariance matrix (inverse of Fisher)
    try:
        Cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        # Singular matrix, return empty ellipse
        return np.array([]), np.array([])

    # Eigenvalue decomposition of covariance
    eigenvalues, eigenvectors = np.linalg.eigh(Cov)

    # Sort by eigenvalues
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Ellipse parameters
    # For n-sigma confidence: chi^2(2 dof, n-sigma) ≈ n^2 for small n
    scale = n_sigma
    width = 2 * scale * np.sqrt(eigenvalues[0])
    height = 2 * scale * np.sqrt(eigenvalues[1])

    # Rotation angle
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Parametric ellipse
    t = np.linspace(0, 2 * np.pi, n_points)
    ellipse_x = (width / 2) * np.cos(t)
    ellipse_y = (height / 2) * np.sin(t)

    # Rotate
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    ellipse = R @ np.vstack([ellipse_x, ellipse_y])

    return ellipse[0, :], ellipse[1, :]


def generate_fisher_contours_plotly(results, theta_fid):
    """
    Generate Plotly HTML for Fisher confidence ellipses.

    Args:
        results: List of result dictionaries with Fisher matrix elements
        theta_fid: Fiducial parameters [theta_A, theta_B]

    Returns:
        HTML string with embedded Plotly plot
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Filter methods to display
    # Only show: Theoretical, MOPED, and best NN (highest log_det_F)
    filtered_results = []

    # Always include Theoretical
    theoretical = next((r for r in results if r['method'] == 'Theoretical'), None)
    if theoretical:
        filtered_results.append(theoretical)

    # Include MOPED
    moped = next((r for r in results if 'MOPED' in r['method']), None)
    if moped:
        filtered_results.append(moped)

    # Find best MLP NN (highest log_det_F with MLP compression)
    nn_methods = [r for r in results if r['method'] not in ['Theoretical', 'MOPED']
                  and r['compression'] != 'None' and 'MLP' in r.get('compression', '')]
    if nn_methods:
        best_nn = max(nn_methods, key=lambda x: x['log_det_F'])
        filtered_results.append(best_nn)

    # Find CNN method (persistence image compression)
    cnn_methods = [r for r in results if 'CNN' in r.get('compression', '')]
    if cnn_methods:
        # Take the first (or best) CNN method
        best_cnn = max(cnn_methods, key=lambda x: x['log_det_F'])
        filtered_results.append(best_cnn)

    # Define colors for different methods
    method_colors = {
        'Theoretical': 'black',
        'MOPED': '#ff7f0e',
        'Best NN': '#1f77b4',
        'CNN': '#2ca02c'
    }

    # Plot ellipses for filtered methods
    for idx, result in enumerate(filtered_results):
        # Determine display category
        if result['method'] == 'Theoretical':
            category = 'Theoretical'
            color = method_colors['Theoretical']
            width = 3
        elif 'MOPED' in result['method']:
            category = 'MOPED'
            color = method_colors['MOPED']
            width = 2
        elif 'CNN' in result.get('compression', ''):
            category = 'CNN'
            color = method_colors['CNN']
            width = 2
        else:
            category = 'Best NN'
            color = method_colors['Best NN']
            width = 2

        dash = 'solid'
        show_in_legend = True

        # Compute only 1-sigma ellipse
        n_sigma = 1
        x, y = compute_fisher_ellipse(
            result['F_00'], result['F_11'], result['F_01'],
            n_sigma=n_sigma
        )

        if len(x) == 0:
            continue

        # Shift to fiducial parameters
        x = x + theta_fid[0]
        y = y + theta_fid[1]

        # Set display name
        if category == 'Best NN':
            display_name = 'Top-K + NN'
            hover_label = f"Top-K + NN ({result['method']})"
        elif category == 'CNN':
            display_name = 'PI + CNN'
            hover_label = f"PI + CNN ({result['method']})"
        else:
            display_name = category
            hover_label = category

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=display_name,
            line=dict(color=color, width=width, dash=dash),
            showlegend=show_in_legend,
            hovertemplate=f"{hover_label}<br>1σ contour<br>" +
                         f"θ_A: %{{x:.3f}}<br>θ_B: %{{y:.3f}}<extra></extra>"
        ))

    # Mark fiducial point
    fig.add_trace(go.Scatter(
        x=[theta_fid[0]], y=[theta_fid[1]],
        mode='markers',
        marker=dict(symbol='cross', size=12, color='red', line=dict(width=2)),
        name='Fiducial θ',
        showlegend=True,
        hovertemplate=f"Fiducial<br>θ_A: {theta_fid[0]}<br>θ_B: {theta_fid[1]}<extra></extra>"
    ))

    # Update layout
    fig.update_layout(
        title="Fisher Information Confidence Contours",
        xaxis_title="A",
        yaxis_title="B",
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        yaxis=dict(scaleanchor="x", scaleratio=1)  # Equal aspect ratio
    )

    # Return as HTML div
    return fig.to_html(include_plotlyjs='cdn', div_id='fisher-contours')


def create_dashboard(json_path="data/results/latest.json", output_path="topofisher/examples/grf/dashboard.html"):
    """
    Generate interactive HTML dashboard from JSON results.

    Args:
        json_path: Path to JSON file with results
        output_path: Path to save HTML file
    """
    # Read JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    results = data['results']

    # Generate Fisher contour plot
    theta_fid = metadata['fisher_config']['theta_fid']
    contour_html = generate_fisher_contours_plotly(results, theta_fid)

    # Generate HTML
    html = generate_html(metadata, results, contour_html)

    # Save HTML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✓ Dashboard created: {output_file}")
    print(f"  Open in browser: file://{output_file.absolute()}")


def generate_html(metadata, results, contour_html):
    """Generate HTML content."""

    # Format delta_theta for display
    delta_theta = metadata['fisher_config']['delta_theta']
    theta_fid = metadata['fisher_config']['theta_fid']

    # Format with up to 3 decimal places, removing trailing zeros
    def format_value(val):
        return f"{val:.3f}".rstrip('0').rstrip('.')

    delta_theta_str = f"[{format_value(delta_theta[0])}, {format_value(delta_theta[1])}]"
    theta_fid_str = f"[{format_value(theta_fid[0])}, {format_value(theta_fid[1])}]"

    # Build table rows
    table_rows = ""
    for r in results:
        table_rows += f"""
        <tr>
            <td>{r['method']}</td>
            <td>{r['filtration']}</td>
            <td>{r['vectorization']}</td>
            <td>{r['compression']}</td>
            <td class="metric">{r['log_det_F']}</td>
            <td class="metric">{r['sigma_A']}</td>
            <td class="metric">{r['sigma_B']}</td>
            <td class="metric">{r['F_00']}</td>
            <td class="metric">{r['F_11']}</td>
            <td class="metric">{r['F_01']}</td>
        </tr>
        """

    # Full HTML with embedded CSS and JavaScript
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TopoFisher Results Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: #f8f9fa;
            border-bottom: 3px solid #28a745;
            color: #333;
            padding: 30px;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        .header-info {{
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .header-info span {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .metadata {{
            background: #f8f9fa;
            padding: 20px 30px;
            border-bottom: 1px solid #e0e0e0;
        }}

        .metadata-toggle {{
            cursor: pointer;
            user-select: none;
            font-weight: 600;
            color: #28a745;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .metadata-toggle:hover {{
            color: #218838;
        }}

        .metadata-content {{
            margin-top: 15px;
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 15px;
        }}

        .metadata-content.hidden {{
            display: none;
        }}

        .metadata-section {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #28a745;
        }}

        .metadata-section h3 {{
            font-size: 1em;
            margin-bottom: 10px;
            color: #333;
        }}

        .metadata-section ul {{
            list-style: none;
            font-size: 0.9em;
            color: #666;
        }}

        .metadata-section li {{
            padding: 3px 0;
        }}

        .visualization-section, .results-section {{
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}

        .section-toggle {{
            cursor: pointer;
            user-select: none;
            font-weight: 600;
            color: #28a745;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 20px 30px;
        }}

        .section-toggle:hover {{
            color: #218838;
            background: #e9ecef;
        }}

        .section-content {{
            padding: 30px;
            background: white;
        }}

        .section-content.hidden {{
            display: none;
        }}

        .table-container {{
            padding: 0;
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}

        thead {{
            background: #28a745;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        th {{
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}

        th:hover {{
            background: #218838;
        }}

        th::after {{
            content: '⇅';
            position: absolute;
            right: 5px;
            opacity: 0.5;
            font-size: 0.8em;
        }}

        th.sorted-asc::after {{
            content: '▲';
            opacity: 1;
        }}

        th.sorted-desc::after {{
            content: '▼';
            opacity: 1;
        }}

        td {{
            padding: 12px 8px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .metric {{
            text-align: right;
            font-family: 'Courier New', monospace;
            font-weight: 500;
        }}

        .best {{
            background: #cfe2ff;
            color: #084298;
            border-left: 3px solid #0d6efd;
        }}

        .worst {{
            background: #fff3cd;
            color: #856404;
            border-left: 3px solid #ffc107;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
            border-top: 1px solid #e0e0e0;
        }}

        .footer a {{
            color: #28a745;
            text-decoration: none;
        }}

        .footer a:hover {{
            text-decoration: underline;
        }}

        @media (max-width: 1200px) {{
            .metadata-content {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}

        @media (max-width: 768px) {{
            .metadata-content {{
                grid-template-columns: 1fr;
            }}

            .header-info {{
                flex-direction: column;
                gap: 10px;
            }}

            .table-container {{
                padding: 15px;
            }}

            table {{
                font-size: 0.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>TopoFisher Results Dashboard</h1>
        </div>

        <div class="metadata">
            <div class="metadata-toggle" onclick="toggleMetadata()">
                <span id="metadata-arrow">▼</span>
                <span>Configuration Details</span>
            </div>
            <div class="metadata-content" id="metadata-content">
                <div class="metadata-section">
                    <h3>Simulator</h3>
                    <ul>
                        <li>Type: {metadata['simulator']['type']}</li>
                        <li>N: {metadata['simulator']['N']}</li>
                        <li>Dimensions: {metadata['simulator']['dim']}</li>
                        <li>Boxlength: {metadata['simulator']['boxlength']}</li>
                    </ul>
                </div>
                <div class="metadata-section">
                    <h3>Filtration</h3>
                    <ul>
                        <li>Type: {metadata['filtration']['type']}</li>
                        <li>Homology Dims: {metadata['filtration']['homology_dimensions']}</li>
                        <li>Min Persistence: {metadata['filtration']['min_persistence']}</li>
                    </ul>
                </div>
                <div class="metadata-section">
                    <h3>Vectorization</h3>
                    <ul>
                        <li>Type: {metadata['vectorization']['type']}</li>
                        <li>Layers: {', '.join(metadata['vectorization']['layers'])}</li>
                        <li>Total Features: {metadata['vectorization']['total_features']}</li>
                    </ul>
                </div>
                <div class="metadata-section">
                    <h3>Fisher Config</h3>
                    <ul>
                        <li>θ_fid: {theta_fid_str}</li>
                        <li>Δθ: {delta_theta_str}</li>
                        <li>n_s: {metadata['fisher_config']['n_s']}</li>
                        <li>n_d: {metadata['fisher_config']['n_d']}</li>
                    </ul>
                </div>
                <div class="metadata-section">
                    <h3>Data Split</h3>
                    <ul>
                        <li>Train: {metadata['data_split']['train']}</li>
                        <li>Validation: {metadata['data_split']['val']}</li>
                        <li>Test: {metadata['data_split']['test']}</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="visualization-section">
            <div class="section-toggle" onclick="toggleSection('contours')">
                <span id="contours-arrow">▼</span>
                <span>Fisher Confidence Contours</span>
            </div>
            <div class="section-content" id="contours-content">
                {contour_html}
            </div>
        </div>

        <div class="results-section">
            <div class="section-toggle" onclick="toggleSection('results')">
                <span id="results-arrow">▼</span>
                <span>Results Table</span>
            </div>
            <div class="section-content" id="results-content">
                <table id="results-table">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">Method</th>
                        <th onclick="sortTable(1)">Filtration</th>
                        <th onclick="sortTable(2)">Vectorization</th>
                        <th onclick="sortTable(3)">Compression</th>
                        <th onclick="sortTable(4)" class="sorted-desc">log det F</th>
                        <th onclick="sortTable(5)">σ(A)</th>
                        <th onclick="sortTable(6)">σ(B)</th>
                        <th onclick="sortTable(7)">F[0,0]</th>
                        <th onclick="sortTable(8)">F[1,1]</th>
                        <th onclick="sortTable(9)">F[0,1]</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            </div>
        </div>

        <div class="footer">
            Generated by TopoFisher • <a href="https://github.com/karthikviswanathn/TopoFisher">GitHub</a>
        </div>
    </div>

    <script>
        // Toggle metadata visibility
        function toggleMetadata() {{
            const content = document.getElementById('metadata-content');
            const arrow = document.getElementById('metadata-arrow');

            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                arrow.textContent = '▼';
            }} else {{
                content.classList.add('hidden');
                arrow.textContent = '▶';
            }}
        }}

        // Toggle section visibility (contours, results)
        function toggleSection(sectionId) {{
            const content = document.getElementById(sectionId + '-content');
            const arrow = document.getElementById(sectionId + '-arrow');

            if (content.classList.contains('hidden')) {{
                content.classList.remove('hidden');
                arrow.textContent = '▼';
            }} else {{
                content.classList.add('hidden');
                arrow.textContent = '▶';
            }}
        }}

        // Table sorting
        let sortColumn = 4;  // Default: log det F
        let sortAscending = false;

        function sortTable(columnIndex) {{
            const table = document.getElementById('results-table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Toggle sort direction if clicking same column
            if (sortColumn === columnIndex) {{
                sortAscending = !sortAscending;
            }} else {{
                sortColumn = columnIndex;
                sortAscending = true;
            }}

            // Sort rows
            rows.sort((a, b) => {{
                const aValue = a.cells[columnIndex].textContent.trim();
                const bValue = b.cells[columnIndex].textContent.trim();

                // Try numeric comparison first
                const aNum = parseFloat(aValue);
                const bNum = parseFloat(bValue);

                let comparison;
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    comparison = aNum - bNum;
                }} else {{
                    comparison = aValue.localeCompare(bValue);
                }}

                return sortAscending ? comparison : -comparison;
            }});

            // Update table
            rows.forEach(row => tbody.appendChild(row));

            // Update header indicators
            const headers = table.querySelectorAll('th');
            headers.forEach((th, i) => {{
                th.classList.remove('sorted-asc', 'sorted-desc');
                if (i === columnIndex) {{
                    th.classList.add(sortAscending ? 'sorted-asc' : 'sorted-desc');
                }}
            }});

            // Highlight best/worst for numeric columns
            if (columnIndex >= 4) {{
                highlightColumn(columnIndex);
            }}
        }}

        function highlightColumn(columnIndex) {{
            const table = document.getElementById('results-table');
            const rows = Array.from(table.querySelectorAll('tbody tr'));

            // Remove existing highlights
            rows.forEach(row => {{
                Array.from(row.cells).forEach(cell => {{
                    cell.classList.remove('best', 'worst');
                }});
            }});

            // Get values (excluding Theoretical row)
            const values = rows
                .filter(row => !row.cells[0].textContent.includes('Theoretical'))
                .map(row => parseFloat(row.cells[columnIndex].textContent))
                .filter(v => !isNaN(v));

            if (values.length === 0) return;

            const max = Math.max(...values);
            const min = Math.min(...values);

            // Highlight cells
            rows.forEach(row => {{
                if (row.cells[0].textContent.includes('Theoretical')) return;

                const value = parseFloat(row.cells[columnIndex].textContent);
                if (isNaN(value)) return;

                // For log det F, higher is better
                // For sigma (constraints), lower is better
                const higherIsBetter = columnIndex === 4;

                if (higherIsBetter) {{
                    if (value === max) row.cells[columnIndex].classList.add('best');
                    if (value === min) row.cells[columnIndex].classList.add('worst');
                }} else if (columnIndex === 5 || columnIndex === 6) {{
                    if (value === min) row.cells[columnIndex].classList.add('best');
                    if (value === max) row.cells[columnIndex].classList.add('worst');
                }}
            }});
        }}

        // Initialize: highlight log det F column
        highlightColumn(4);
    </script>
</body>
</html>
"""

    return html


if __name__ == "__main__":
    create_dashboard()
