# app.py
import polars as pl
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from shiny import App, ui, render, reactive

# Helper functions
def clean_currency(value):
    if isinstance(value, str):
        return value.replace('$', '').replace(',', '')
    return value

def process_row(row, G):
    from_addr = row['From']
    to_addr = row['To']
    
    # Clean and convert Amount and Value to float
    try:
        amount = float(clean_currency(row['Amount'])) if row['Amount'] is not None else 0
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert Amount '{row['Amount']}' to float: {e}")
        amount = 0
        
    try:
        value_usd = float(clean_currency(row['Value (USD)'])) if row['Value (USD)'] is not None else 0
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert Value '{row['Value (USD)']}' to float: {e}")
        value_usd = 0
    
    # Add nodes with attributes
    if from_addr not in G:
        G.add_node(from_addr, 
                  label=row['From'],
                  total_sent=0,
                  total_received=0,
                  transaction_count_out=0,
                  transaction_count_in=0)
    
    if to_addr not in G:
        G.add_node(to_addr, 
                  label=row['To'],
                  total_sent=0,
                  total_received=0,
                  transaction_count_out=0,
                  transaction_count_in=0)
    
    # Update node metrics
    G.nodes[from_addr]['total_sent'] += value_usd
    G.nodes[from_addr]['transaction_count_out'] += 1
    G.nodes[to_addr]['total_received'] += value_usd
    G.nodes[to_addr]['transaction_count_in'] += 1
    
    # Add or update edge
    if G.has_edge(from_addr, to_addr):
        G[from_addr][to_addr]['weight'] += 1
        G[from_addr][to_addr]['total_amount'] += amount
        G[from_addr][to_addr]['total_value_usd'] += value_usd
    else:
        G.add_edge(from_addr, to_addr, 
                  weight=1, 
                  total_amount=amount,
                  total_value_usd=value_usd)
    
    return G

def create_node_trace(G, pos, driver_scores, top_origin_nodes, size_by='Driver Score'):
    nodes = list(G.nodes())
    
    # Get the appropriate node attribute for sizing
    if size_by == 'Driver Score':
        values = [driver_scores.get(node, 0) for node in nodes]
    elif size_by == 'TX Out':
        values = [G.nodes[node]['transaction_count_out'] for node in nodes]
    elif size_by == 'TX In':
        values = [G.nodes[node]['transaction_count_in'] for node in nodes]
    elif size_by == 'Total Sent':
        values = [G.nodes[node]['total_sent'] for node in nodes]
    elif size_by == 'Total Received':
        values = [G.nodes[node]['total_received'] for node in nodes]
    
    # Normalize sizes relative to maximum (if there's a non-zero max)
    max_value = max(values) if max(values) > 0 else 1
    # Scale between 5 (minimum) and 40 (maximum) for visibility
    sizes = [5 + (35 * (value / max_value)) for value in values]
    
    # Create node colors based on out/in ratio
    colors = []
    for node in nodes:
        in_count = G.nodes[node]['transaction_count_in']
        out_count = G.nodes[node]['transaction_count_out']
        ratio = out_count / max(in_count, 1)
        
        if node in top_origin_nodes:
            colors.append('#e63946')  # Red for top origins
        elif ratio > 1.5:
            colors.append('#f9844a')  # Orange for senders
        elif 0.5 < ratio <= 1.5:
            colors.append('#2a9d8f')  # Teal for balanced
        else:
            colors.append('#4361ee')  # Blue for receivers
    
    # Create the updated node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in nodes],
        y=[pos[node][1] for node in nodes],
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(width=1, color='rgba(250,250,250,0.3)')
        ),
        text=[f"Address: {G.nodes[node]['label']}<br>TX Out: {G.nodes[node]['transaction_count_out']}<br>" +
              f"TX In: {G.nodes[node]['transaction_count_in']}<br>" +
              f"Total Sent: ${G.nodes[node]['total_sent']:.2f}<br>" +
              f"Total Received: ${G.nodes[node]['total_received']:.2f}<br>" +
              f"Driver Score: {driver_scores.get(node, 0):.2f}" for node in nodes],
        hoverinfo='text',
        showlegend=False
    )
    
    return node_trace

def generate_network_figure(G, pos, driver_scores, top_origin_nodes, size_by='Driver Score'):
    # Set color palette for light theme
    colors = {
        'background': '#ffffff',
        'text': '#455a64',
        'grid': '#e0e0e0',
        'origin_node': '#e63946',  # Red for top origins
        'sender_node': '#f9844a',  # Orange for senders
        'balanced_node': '#2a9d8f', # Teal for balanced
        'receiver_node': '#4361ee', # Blue for receivers
        'edge': '#adb5bd',
    }
    
    # Create figure
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter"}]]
    )
    
    # Add edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_weight = G[edge[0]][edge[1]]['weight']
        total_value = G[edge[0]][edge[1]]['total_value_usd']
        total_amount = G[edge[0]][edge[1]]['total_amount']
        
        # Width based on transaction count
        width = 1 + (edge_weight / 5)  # Scale edge width
        
        # Color based on source node being a top origin
        if edge[0] in top_origin_nodes:
            color = 'rgba(230, 57, 70, 0.5)'  # Red for edges from top origins
        else:
            color = 'rgba(173, 181, 189, 0.5)'  # Gray for normal edges
        
        # Create the main edge line
        edge_trace = go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"From: {G.nodes[edge[0]]['label']}<br>" + \
                 f"To: {G.nodes[edge[1]]['label']}<br>" + \
                 f"Transactions: {edge_weight}<br>" + \
                 f"Total Amount: {total_amount}<br>" + \
                 f"Total Value: ${total_value:.2f}",
            mode='lines',
            showlegend=False
        )
        
        fig.add_trace(edge_trace)
    
    # Add nodes
    node_trace = create_node_trace(G, pos, driver_scores, top_origin_nodes, size_by)
    fig.add_trace(node_trace)
    
    # Add legend entries
    legend_entries = [
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors['origin_node']),
            name='Top Origin Wallets'
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors['sender_node']),
            name='Sender Wallets'
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors['balanced_node']),
            name='Balanced Wallets'
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=colors['receiver_node']),
            name='Receiver Wallets'
        )
    ]
    
    for entry in legend_entries:
        fig.add_trace(entry)
    
    # Update layout for light theme
    fig.update_layout(
        title="Token Transaction Network Visualization",
        title_font=dict(size=20, color=colors['text']),
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1,
            font=dict(color=colors['text']),
            itemsizing='constant'
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=50),
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font=dict(family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif", color=colors['text']),
        height=700,
    )
    
    # Set axis properties
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    return fig

# Define UI with a harmonized light theme
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css"),
        ui.tags.script(src="https://code.jquery.com/jquery-3.6.0.min.js"),
        ui.tags.script(src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js")
    ),
    ui.tags.style("""
        body {
            background-color: #f8f9fa;
            color: #212529;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Fix layout to prevent collisions */
        .shiny-split-layout > div:first-child {
            flex: 0 0 250px !important;
            width: 250px !important;
            max-width: 250px !important;
        }
        .shiny-split-layout > div:last-child {
            margin-left: 20px;
            flex: 1;
        }
        .sidebar {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            width: 100%;
            box-sizing: border-box;
        }
        .sidebar table {
            table-layout: fixed;
            width: 100%;
            word-wrap: break-word;
        }
        .main-panel {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            width: 100%;
        }
        .explanation-panel {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stats-panel {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .app-title {
            color: #3f51b5;
            margin-bottom: 20px;
        }
        h1, h2, h3 {
            color: #3f51b5;
        }
        h4 {
            color: #455a64;
        }
        .section-header {
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .irs-bar, .irs-bar-edge {
            background: #3f51b5;
            border-color: #3f51b5;
        }
        .irs-handle {
            border-color: #3f51b5;
        }
        .form-control {
            border-color: #ddd;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background-color: #f0f2f5;
            color: #455a64;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        /* Custom DataTable styling */
        .dataTable {
            width: 100%;
            margin-bottom: 1rem;
            color: #212529;
        }
        .dataTable thead th {
            background-color: #f0f2f5;
            color: #455a64;
            border-bottom: 2px solid #ddd;
            cursor: pointer;
            padding: 0.75rem;
            vertical-align: bottom;
        }
        .dataTable tbody td {
            padding: 0.75rem;
            vertical-align: top;
            border-bottom: 1px solid #eee;
        }
        .dataTable tbody tr:hover {
            background-color: rgba(0,0,0,.05);
        }
        /* Ensure table content in sidebar doesn't overflow */
        .sidebar td, .sidebar th {
            word-break: break-word;
            overflow-wrap: break-word;
            font-size: 0.9rem;
            padding: 6px;
        }
        /* Stats table styling */
        .stats-table {
            width: 100%;
            border-collapse: collapse;
        }
        .stats-table th, .stats-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .stats-table th {
            background-color: #f0f2f5;
            font-weight: bold;
        }
    """),
    ui.tags.div(
        ui.tags.h1("Token Transaction Network Analyzer", class_="app-title"),
        class_="container-fluid"
    ),
    # Use a fixed-width layout with proper spacing
    ui.layout_sidebar(
        ui.sidebar(
            ui.tags.div(
                ui.tags.h4("Visualization Options", class_="section-header"),
                ui.input_file("file_upload", "Upload CSV File:", multiple=False),
                ui.input_select(
                    "size_by", 
                    "Size Nodes By:", 
                    {"Driver Score": "Driver Score", 
                     "TX Out": "TX Out", 
                     "TX In": "TX In", 
                     "Total Sent": "Total Sent", 
                     "Total Received": "Total Received"}
                ),
                ui.input_slider("top_wallets", "Top Wallets to Display:", 5, 20, 10),
                class_="sidebar"
            ),
            width=250  # Fixed width for sidebar
        ),
        ui.tags.div(
            ui.tags.div(
                ui.tags.h2("Token Transaction Network", class_="section-header"),
                ui.output_ui("network_plot"),
                class_="main-panel"
            ),
            ui.tags.div(
                ui.tags.h2("Top Driver Wallets", class_="section-header"),
                ui.tags.div(
                    ui.output_ui("wallet_table_ui"),
                    id="wallet-table-container"
                ),
                class_="main-panel"
            ),
            # New side-by-side layout for explanation and stats
            ui.row(
                ui.column(6,
                    ui.tags.div(
                        ui.tags.div(
                            ui.tags.h3("Understanding Wallet Types", class_="section-header"),
                            ui.tags.p("Each wallet type in the visualization represents a distinct role in the token transaction network:"),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.strong("Top Origin Wallets"), " - These are the most influential wallets that initiate a high volume of transactions."),
                                ui.tags.li(ui.tags.strong("Sender Wallets"), " - These wallets primarily send tokens to other addresses rather than receiving them."),
                                ui.tags.li(ui.tags.strong("Balanced Wallets"), " - These wallets have roughly equal sending and receiving activity."),
                                ui.tags.li(ui.tags.strong("Receiver Wallets"), " - These wallets primarily receive tokens rather than sending them."),
                            ),
                        ),
                        ui.tags.div(
                            ui.tags.h3("Driver Score Explanation", class_="section-header"),
                            ui.tags.p("The Driver Score is a composite metric that quantifies a wallet's influence within the token transaction network:"),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.strong("Definition"), ": A weighted calculation that identifies wallets initiating significant transaction activity"),
                                ui.tags.li(ui.tags.strong("Components"), ": Combines outgoing transaction count, total value sent, and network centrality metrics"),
                                ui.tags.li(ui.tags.strong("Purpose"), ": Identifies wallets that play a central role in driving network activity and fund flows"),
                                ui.tags.li(ui.tags.strong("Calculation"), ": Weights outgoing transactions (50%), total value sent (30%), and network position (20%)"),
                            ),
                        ),
                        class_="explanation-panel"
                    )
                ),
                ui.column(6,
                    ui.tags.div(
                        ui.tags.h3("Network Statistics", class_="section-header"),
                        ui.output_table("stats_table", class_="stats-table"),
                        class_="stats-panel"
                    )
                )
            )
        )
    ),
    # JavaScript for table sorting functionality
    ui.tags.script("""
    $(document).ready(function() {
        // Function to initialize DataTable
        function initDataTable() {
            if ($.fn.DataTable && $('#wallet-table').length) {
                // Destroy existing table if needed
                if ($.fn.DataTable.isDataTable('#wallet-table')) {
                    $('#wallet-table').DataTable().destroy();
                }
                
                // Initialize with options
                $('#wallet-table').DataTable({
                    paging: false,
                    searching: false,
                    info: false,
                    order: [[7, 'desc']], // Sort by Driver Score by default
                    columnDefs: [
                        { type: 'num', targets: [1, 2, 3, 4, 5, 6, 7] } // Numeric columns
                    ]
                });
            }
        }
        
        // Try to initialize when document is ready
        setTimeout(initDataTable, 1000);
        
        // Initialize when Shiny loads new content
        $(document).on('shiny:value', function(event) {
            if (event.name === 'wallet_table_ui') {
                setTimeout(initDataTable, 500);
            }
        });
        
        // Re-initialize when the DOM changes
        const observer = new MutationObserver(function(mutations) {
            for (let mutation of mutations) {
                if (mutation.type === 'childList' && 
                    mutation.target.id === 'wallet-table-container') {
                    setTimeout(initDataTable, 100);
                }
            }
        });
        
        // Start observing the wallet table container
        if (document.getElementById('wallet-table-container')) {
            observer.observe(document.getElementById('wallet-table-container'), {
                childList: true,
                subtree: true
            });
        }
    });
    """)
)

# Server function
def server(input, output, session):
    # Initialize reactive values
    df_reactive = reactive.Value(None)
    graph_reactive = reactive.Value(None)
    pos_reactive = reactive.Value(None)
    driver_scores_reactive = reactive.Value(None)
    top_origin_nodes_reactive = reactive.Value(None)
    wallet_df_reactive = reactive.Value(None)
    stats_df_reactive = reactive.Value(None)
    
    @reactive.Effect
    @reactive.event(input.file_upload)
    def upload_file():
        file_info = input.file_upload()
        if file_info is not None and len(file_info) > 0:
            file_path = file_info[0]["datapath"]
            try:
                # Read data with polars
                df = pl.read_csv(file_path, truncate_ragged_lines=True)
                df_reactive.set(df)
                
                # Process the data to create graph
                G = nx.DiGraph()
                for row in df.iter_rows(named=True):
                    G = process_row(row, G)
                
                # Calculate positions
                pos = nx.kamada_kawai_layout(G)
                pos_reactive.set(pos)
                
                # Calculate metrics
                outgoing_counts = {node: G.nodes[node]['transaction_count_out'] for node in G.nodes()}
                top_origins = sorted(outgoing_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                top_origin_nodes = [node for node, _ in top_origins]
                top_origin_nodes_reactive.set(top_origin_nodes)
                
                # Calculate centrality metrics
                betweenness_centrality = nx.betweenness_centrality(G)
                out_degree_centrality = nx.out_degree_centrality(G)
                
                # Calculate driver scores
                driver_scores = {}
                for node in G.nodes():
                    driver_scores[node] = (
                        G.nodes[node]['transaction_count_out'] * 0.5 +
                        G.nodes[node]['total_sent'] * 0.3 +
                        betweenness_centrality[node] * 100 +
                        out_degree_centrality[node] * 100
                    )
                driver_scores_reactive.set(driver_scores)
                
                # Store the graph
                graph_reactive.set(G)
                
                # Create summary tables
                wallet_summary = []
                for node in G.nodes():
                    in_count = G.nodes[node]['transaction_count_in']
                    out_count = G.nodes[node]['transaction_count_out']
                    
                    wallet_summary.append({
                        'Address': G.nodes[node]['label'],
                        'TX Out': out_count,
                        'TX In': in_count,
                        'Out/In Ratio': round(out_count / max(in_count, 1), 2),
                        'Total TX': out_count + in_count,
                        'Total Sent ($)': round(G.nodes[node]['total_sent'], 2),
                        'Total Received ($)': round(G.nodes[node]['total_received'], 2),
                        'Driver Score': round(driver_scores[node], 2)
                    })
                
                wallet_df = pl.DataFrame(wallet_summary)
                wallet_df = wallet_df.sort("Driver Score", descending=True)
                wallet_df_reactive.set(wallet_df)
                
                # Network stats
                stats_df = pl.DataFrame([
                    {"Metric": "Total Transactions", "Value": df.height},
                    {"Metric": "Total Unique Wallets", "Value": len(G.nodes())},
                    {"Metric": "Top Origin Wallet", "Value": G.nodes[top_origin_nodes[0]]['label'] if top_origin_nodes else "None"},
                    {"Metric": "Most Active Origin TX Count", "Value": top_origins[0][1] if top_origins else 0},
                    {"Metric": "Network Density", "Value": round(nx.density(G), 4)},
                ])
                stats_df_reactive.set(stats_df)
                
            except Exception as e:
                print(f"Error processing file: {e}")
                try:
                    ui.notification_show(f"Error: {e}", type="error")
                except:
                    print(f"Could not display notification: {e}")
    
    @output
    @render.ui
    def network_plot():
        G = graph_reactive.get()
        pos = pos_reactive.get()
        driver_scores = driver_scores_reactive.get()
        top_origin_nodes = top_origin_nodes_reactive.get()
        
        if G is None or pos is None or driver_scores is None or top_origin_nodes is None:
            empty_fig = go.Figure().update_layout(
                title="Upload a CSV file to display the network",
                height=400,
                paper_bgcolor='#ffffff',
                plot_bgcolor='#ffffff',
                font=dict(color='#455a64')
            )
            return ui.HTML(empty_fig.to_html(include_plotlyjs="cdn"))
        
        fig = generate_network_figure(G, pos, driver_scores, top_origin_nodes, input.size_by())
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def wallet_table_ui():
        wallet_df = wallet_df_reactive.get()
        if wallet_df is None:
            return ui.HTML("<p>Upload data to view wallet information</p>")
        
        # Get top wallets based on slider
        top_wallet_df = wallet_df.head(input.top_wallets())
        
        # Fix the HTML to properly initialize with DataTables
        html_table = '<table id="wallet-table" class="display stripe hover">'
        
        # Add header
        html_table += '<thead><tr>'
        for col in top_wallet_df.columns:
            # Determine data type for proper sorting
            data_type = "data-type=\"string\""
            if col in ['TX Out', 'TX In', 'Out/In Ratio', 'Total TX', 'Total Sent ($)', 'Total Received ($)', 'Driver Score']:
                data_type = "data-type=\"numeric\""
            
            html_table += f'<th {data_type}>{col}</th>'
        html_table += '</tr></thead>'
        
        # Add body
        html_table += '<tbody>'
        for row in top_wallet_df.iter_rows(named=True):
            html_table += '<tr>'
            for col in top_wallet_df.columns:
                # Format cell data based on column type
                if col == 'Address':
                    # Handle address column
                    cell_value = row[col]
                else:
                    # Handle numeric columns
                    cell_value = row[col]
                
                html_table += f'<td>{cell_value}</td>'
            html_table += '</tr>'
        html_table += '</tbody>'
        
        html_table += '</table>'
        
        return ui.HTML(html_table)
    
    @output
    @render.table
    def stats_table():
        stats_df = stats_df_reactive.get()
        if stats_df is None:
            # Return an empty table with header
            empty_df = pl.DataFrame([
                {"Metric": "Total Transactions", "Value": "-"},
                {"Metric": "Total Unique Wallets", "Value": "-"},
                {"Metric": "Top Origin Wallet", "Value": "-"},
                {"Metric": "Most Active Origin TX Count", "Value": "-"},
                {"Metric": "Network Density", "Value": "-"}
            ])
            return empty_df
        
        return stats_df

# Run the application
app = App(app_ui, server)