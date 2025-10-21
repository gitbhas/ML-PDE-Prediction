import json
import urllib3
import pandas as pd
import numpy as np
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def handler(event, context):
    # Using packages from the layer
    df = pd.DataFrame({'A': np.random.rand(5), 'B': np.random.rand(5)})
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Hello from Containerized Lambda with Layers!',
            'dataframe_info': df.to_dict(),
        })
    }