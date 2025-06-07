import time
import requests


class RateLimitedSession:
    def __init__(self, max_requests_per_minute, max_retries=3, retry_delay=1):
        self.max_requests = max_requests_per_minute
        self.interval = 60.0 / max_requests_per_minute
        self.last_request_time = 0
        self.session = requests.Session()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _wait(self):
        now = time.time()
        elapsed = now - self.last_request_time
        wait_time = self.interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def _request_with_retries(self, method, url, **kwargs):
        for attempt in range(self.max_retries):
            self._wait()
            response = self.session.request(method, url, **kwargs)
            
            if response.status_code == 200:
                return response
            else:
                # print(f"[Attempt {attempt + 1}] Non-200 response: {response.status_code}")
                time.sleep(self.retry_delay)
        
        raise Exception(f"Failed to get a 200 response after {self.max_retries} attempts")
    
    def get(self, url, **kwargs):
        return self._request_with_retries('GET', url, **kwargs)
    
    def post(self, url, **kwargs):
        return self._request_with_retries('POST', url, **kwargs)


class ApiConnector:
    def __init__(self, max_requests_per_minute=200 * 60):
        self.api = RateLimitedSession(max_requests_per_minute=max_requests_per_minute)

    def fetch_sequence(self, protein_acc):
        url = f"https://www.ebi.ac.uk/interpro/api/protein/uniprot/{protein_acc}/"
        try:
            response = self.api.get(url)
            protein_info = response.json()
        except Exception as e:
            print(f"Error fetching data for {protein_acc}: {e}")
            return None
        sequence = protein_info["metadata"]["sequence"]
        return sequence
