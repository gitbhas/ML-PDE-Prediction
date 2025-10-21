import json
import os

def lambda_handler(event, context):
    """Debug handler to check what files are available"""
    
    files = []
    for root, dirs, filenames in os.walk('/var/task'):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({
            'files': files,
            'cwd': os.getcwd(),
            'env': dict(os.environ)
        }, indent=2)
    }