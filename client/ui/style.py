STYLESHEET = """
    QMainWindow {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    QLabel { color: #eee; font-family: 'Segoe UI', Arial; }
    
    #HeaderLabel {
        font-size: 24px;
        font-weight: bold;
        color: #ddd;
        padding: 10px;
        border-bottom: 2px solid #333;
    }
    
    #FeedFrame {
        background-color: #252525;
        border: 2px solid #444;
        border-radius: 10px;
    }
    
    #FeedTitle {
        font-size: 16px;
        font-weight: bold;
        color: #aaa;
        margin-bottom: 5px;
    }
    
    /* Alert Bar Styles */
    #AlertBar_Safe {
        background-color: #154215; /* Dark Green */
        color: #00FF00;
        font-size: 28px;
        font-weight: bold;
        border-radius: 5px;
        border: 1px solid #00FF00;
    }
    #AlertBar_Warning {
        background-color: #58390b; /* Dark Orange */
        color: #FFAA00;
        font-size: 28px;
        font-weight: bold;
        border-radius: 5px;
        border: 1px solid #FFAA00;
    }
    #AlertBar_Critical {
        background-color: #4a0d0d; /* Dark Red */
        color: #FF0000;
        font-size: 32px; /* Bigger font */
        font-weight: 900;
        border-radius: 5px;
        border: 2px solid #FF0000;
        animation: blink 1s infinite; /* Advanced QT animation supported? Mostly no, but static looks good */
    }

    /* Buttons */
    QPushButton {
        background-color: #333;
        border: 1px solid #555;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: bold;
        color: white;
    }
    QPushButton:hover { background-color: #444; }
    
    #BtnStart { background-color: #333; border: 1px solid #555; }
    #BtnStart:hover { background-color: #4a4a4a; }
    
    #BtnStop { background-color: #2b2b2b; border: 1px solid #444; }
    #BtnStop:hover { background-color: #3d3d3d; }
    
    #BtnExit { background-color: #222; border: 1px solid #666; color: #aaa; }
    #BtnExit:hover { background-color: #444; color: white; }
    
    QComboBox {
        background-color: #333;
        color: white;
        border: 1px solid #555;
        padding: 8px;
        border-radius: 5px;
    }
"""