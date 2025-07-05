import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional, Dict

class APIKeyRotator:
    def __init__(self):
        """
        Secure API key rotator with improved security practices.
        
        Precondition:
            - API keys will be read from an environment variable API_KEYS,
              which should be a comma-separated list of key strings.
        Postcondition:
            - If no .env file exists, one is created by prompting the user.
            - Keys are validated and usage is tracked securely.
        """
        # Set up secure logging
        self.logger = logging.getLogger(__name__)
        
        # Load keys securely
        self.keys = self._load_keys()
        self._index = 0
        self._key_usage = {key: 0 for key in self.keys}
        
        print(f"Initialized API key rotator with {len(self.keys)} keys")

    def _load_keys(self) -> List[str]:
        """
        Load API keys from environment variables with validation.
        
        Returns:
            List of validated API keys
            
        Raises:
            ValueError: If no valid keys are found
        """
        env_path = Path(".env")
        if not env_path.exists():
            self._create_env_file(env_path)

        # Load from .env (or from real environment if already set)
        load_dotenv(dotenv_path=env_path)

        keys_str = os.getenv("API_KEYS", "")
        if not keys_str:
            raise ValueError("No API_KEYS found in the environment. Please set API_KEYS in .env")

        # Split on commas and strip whitespace
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not keys:
            raise ValueError("Found API_KEYS but no valid keys were parsed.")

        # Validate key format (basic check)
        for key in keys:
            if len(key) < 20:  # Basic length validation for API keys
                raise ValueError(f"API key appears invalid (too short): {key[:10]}...")
            if not key.isalnum() and not any(c in key for c in ['-', '_']):
                raise ValueError(f"API key contains invalid characters: {key[:10]}...")

        return keys

    def _create_env_file(self, env_path: Path):
        """
        Prompt the user to enter API keys in sequence (key1, key2, ...)
        until typing 'exit', then write them into a .env file as:
            API_KEYS=key1,key2,...
        """
        print(f"'{env_path}' not found. Creating it with your API keys.")
        keys = []
        i = 1
        while True:
            prompt = f"Enter API key {i} (or type 'exit' to finish): "
            key = input(prompt).strip()
            if key.lower() == 'exit':
                if keys:
                    break
                print("You must enter at least one API key to continue.")
                continue
            if not key:
                print("Empty input. Please enter a valid API key or type 'exit' if done.")
                continue
            
            # Basic validation during input
            if len(key) < 20:
                print("API key appears too short. Please check and re-enter.")
                continue
                
            keys.append(key)
            i += 1

        # Write out the .env file
        try:
            with open(env_path, 'w') as f:
                f.write(f"API_KEYS={','.join(keys)}\n")
            print(f"Created '{env_path}' with {len(keys)} key(s).")
        except PermissionError:
            raise PermissionError(f"Cannot write to {env_path}. Check file permissions.")
        except Exception as e:
            raise Exception(f"Error creating .env file: {e}")

    def current_key(self) -> str:
        """
        Return the API key at the current position.
        
        Returns:
            Current API key string
        """
        return self.keys[self._index]

    def rotate_key(self) -> str:
        """
        Advance to the next key (wrapping around) and return it.
        Logs rotation without exposing key information.
        
        Returns:
            New API key string
        """
        self._index = (self._index + 1) % len(self.keys)
        new_key = self.current_key()
        
        # Track usage for the previous key
        if self._index > 0:
            prev_key = self.keys[self._index - 1]
        else:
            prev_key = self.keys[-1]
        self._key_usage[prev_key] += 1
        
        # Log rotation without exposing key details
        print(f"API key rotated (key {self._index + 1} of {len(self.keys)})")
        return new_key

    def get_key_usage_stats(self) -> Dict:
        """
        Get usage statistics for monitoring without exposing actual keys.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_keys": len(self.keys),
            "current_key_index": self._index,
            "usage_counts": {f"key_{i}": count for i, count in enumerate(self._key_usage.values())},
            "total_requests": sum(self._key_usage.values())
        }

    def get_quota_status(self) -> Dict:
        """
        Get current quota status for monitoring.
        
        Returns:
            Dictionary with quota information
        """
        return {
            "current_key": f"key_{self._index + 1}",
            "total_keys": len(self.keys),
            "requests_on_current_key": self._key_usage.get(self.current_key(), 0)
        }