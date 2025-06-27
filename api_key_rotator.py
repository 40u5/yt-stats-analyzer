import os
from pathlib import Path
from dotenv import load_dotenv

class APIKeyRotator:
    def __init__(self):
        """
        Precondition:
            - API keys will be read from an environment variable API_KEYS,
              which should be a comma-separated list of key strings.
        Postcondition:
            - If no .env file exists, one is created by prompting the user.
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
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.keys:
            raise ValueError("Found API_KEYS but no valid keys were parsed.")

        self._index = 0

    def _create_env_file(self, env_path: Path):
        """
        Prompt the user to enter API keys in sequence (key1, key2, ...)
        until typing 'exit', then write them into a .env file as:
            API_KEYS=key1,key2,...
        """
        print(f"[INFO] '{env_path}' not found. Let's create it with your API keys.")
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
            keys.append(key)
            i += 1

        # Write out the .env file
        with open(env_path, 'w') as f:
            f.write(f"API_KEYS={','.join(keys)}\n")
        print(f"[INFO] Created '{env_path}' with {len(keys)} key(s).")

    def current_key(self):
        """Return the API key at the current position."""
        return self.keys[self._index]

    def rotate_key(self):
        """Advance to the next key (wrapping around) and return it."""
        self._index = (self._index + 1) % len(self.keys)
        new_key = self.current_key()
        print(f"[INFO] Rotated to key #{self._index}")
        return new_key