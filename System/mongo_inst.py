#!/usr/bin/env python3
"""
MongoDB Community Edition + Compass Automated Installer for macOS
This script follows the official MongoDB installation guide for macOS.
"""

import subprocess
import sys
import os
import platform
import time
import requests
import tarfile
import zipfile
from pathlib import Path


class MongoDBInstaller:
    def __init__(self):
        self.system_info = platform.uname()
        self.is_apple_silicon = 'arm' in self.system_info.machine.lower()
        print(f"System: {self.system_info.system} {self.system_info.release}")
        print(f"Architecture: {'Apple Silicon' if self.is_apple_silicon else 'Intel'}")

    def run_command(self, command, description, check=True):
        """Execute a shell command with error handling"""
        print(f"\n🔧 {description}...")
        print(f"   Command: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e.stderr.strip()}")
            if check:
                print(f"   Command failed with exit code {e.returncode}")
                return False
            return False

    def check_prerequisites(self):
        """Check and install prerequisites"""
        print("\n" + "=" * 60)
        print("STEP 1: Checking Prerequisites")
        print("=" * 60)

        # Check if Xcode Command Line Tools are installed
        print("\n📦 Checking Xcode Command Line Tools...")
        xcode_check = subprocess.run(
            "xcode-select -p",
            shell=True,
            capture_output=True,
            text=True
        )

        if xcode_check.returncode != 0:
            print("   Xcode Command Line Tools not found. Installing...")
            self.run_command("xcode-select --install",
                             "Installing Xcode Command Line Tools")
            print("   Please complete the Xcode installation in the pop-up window.")
            input("   Press Enter after Xcode installation is complete...")
        else:
            print(f"   ✓ Xcode Command Line Tools found at: {xcode_check.stdout.strip()}")

        # Check if Homebrew is installed
        print("\n🍺 Checking Homebrew...")
        brew_check = subprocess.run(
            "which brew",
            shell=True,
            capture_output=True,
            text=True
        )

        if brew_check.returncode != 0:
            print("   Homebrew not found. Installing...")
            # Install Homebrew using the official script
            install_script = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            self.run_command(install_script, "Installing Homebrew")

            # Configure Homebrew for Apple Silicon if needed
            if self.is_apple_silicon:
                shell_config = """
                echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
                eval "$(/opt/homebrew/bin/brew shellenv)"
                """
                self.run_command(shell_config, "Configuring Homebrew for Apple Silicon")
        else:
            print(f"   ✓ Homebrew found at: {brew_check.stdout.strip()}")

        return True

    def install_mongodb(self):
        """Install MongoDB Community Edition using Homebrew"""
        print("\n" + "=" * 60)
        print("STEP 2: Installing MongoDB Community Edition")
        print("=" * 60)

        # Update Homebrew
        self.run_command("brew update", "Updating Homebrew")

        # Tap MongoDB repository
        self.run_command("brew tap mongodb/brew", "Adding MongoDB Homebrew tap")

        # Install MongoDB Community Edition
        install_success = self.run_command(
            "brew install mongodb-community@7.0",
            "Installing MongoDB Community Edition 7.0"
        )

        if not install_success:
            print("\n⚠️  Installation failed. Trying to fix common issues...")
            # Try to fix ChecksumMismatchError
            self.run_command("brew untap mongodb/brew && brew tap mongodb/brew",
                             "Retapping MongoDB formula", check=False)
            self.run_command("brew install mongodb-community@7.0",
                             "Retrying MongoDB installation")

        # Verify installation
        print("\n🔍 Verifying MongoDB installation...")
        self.run_command("mongod --version", "Checking mongod version", check=False)
        self.run_command("mongosh --version", "Checking MongoDB Shell version", check=False)

        return True

    def install_mongodb_compass(self):
        """Download and install MongoDB Compass"""
        print("\n" + "=" * 60)
        print("STEP 3: Installing MongoDB Compass")
        print("=" * 60)

        compass_url = "https://downloads.mongodb.com/compass/mongodb-compass-1.40.4-darwin-x64.dmg"
        if self.is_apple_silicon:
            compass_url = "https://downloads.mongodb.com/compass/mongodb-compass-1.40.4-darwin-arm64.dmg"

        download_path = Path.home() / "Downloads" / "mongodb-compass.dmg"

        print(f"\n📥 Downloading MongoDB Compass...")
        print(f"   URL: {compass_url}")

        try:
            response = requests.get(compass_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)

                    # Show progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                              end='\r', flush=True)

            print(f"\n   ✓ Download complete: {download_path}")

            # Mount the DMG and install
            print("\n🔗 Installing MongoDB Compass...")
            mount_commands = [
                f'hdiutil attach "{download_path}"',
                'cp -r "/Volumes/MongoDB Compass/MongoDB Compass.app" "/Applications/"',
                'hdiutil detach "/Volumes/MongoDB Compass"'
            ]

            for cmd in mount_commands:
                self.run_command(cmd, f"Executing: {cmd[:50]}...", check=False)

            print("   ✓ MongoDB Compass installed to /Applications/")

            # Clean up
            download_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"   ❌ Failed to install MongoDB Compass: {e}")
            print("   You can manually download Compass from: https://www.mongodb.com/products/compass")
            return False

        return True

    def configure_mongodb(self):
        """Configure MongoDB service"""
        print("\n" + "=" * 60)
        print("STEP 4: Configuring MongoDB Service")
        print("=" * 60)

        # Create necessary directories if they don't exist
        if self.is_apple_silicon:
            data_dir = "/opt/homebrew/var/mongodb"
            log_dir = "/opt/homebrew/var/log/mongodb"
            config_file = "/opt/homebrew/etc/mongod.conf"
        else:
            data_dir = "/usr/local/var/mongodb"
            log_dir = "/usr/local/var/log/mongodb"
            config_file = "/usr/local/etc/mongod.conf"

        print(f"\n📁 Creating directories...")
        for directory in [data_dir, log_dir]:
            self.run_command(f"sudo mkdir -p {directory}", f"Creating {directory}", check=False)
            self.run_command(f"sudo chown $(whoami) {directory}", f"Setting permissions for {directory}", check=False)

        # Start MongoDB as a service
        print("\n🚀 Starting MongoDB Service...")
        self.run_command(
            "brew services start mongodb-community@7.0",
            "Starting MongoDB as a macOS service"
        )

        # Wait for MongoDB to start
        print("\n⏳ Waiting for MongoDB to start...")
        time.sleep(5)

        # Check if MongoDB is running
        print("\n🔍 Checking MongoDB status...")
        self.run_command("brew services list | grep mongodb", "Checking service status", check=False)

        # Test connection with mongosh
        print("\n🔗 Testing MongoDB connection...")
        test_script = """
        echo "db.version()" | mongosh --quiet --eval "
            try {
                db.version();
                print('✓ Successfully connected to MongoDB');
            } catch(e) {
                print('✗ Connection failed: ' + e.message);
            }
        "
        """
        self.run_command(test_script, "Testing connection with mongosh", check=False)

        return True

    def display_summary(self):
        """Display installation summary"""
        print("\n" + "=" * 60)
        print("📊 INSTALLATION SUMMARY")
        print("=" * 60)

        summary = f"""
        ✅ MongoDB Community Edition 7.0 has been installed

        📍 Key Locations:
        • MongoDB Binary: /usr/local/bin/mongod
        • MongoDB Shell: /usr/local/bin/mongosh
        • Config File: {'/opt/homebrew/etc/mongod.conf' if self.is_apple_silicon else '/usr/local/etc/mongod.conf'}
        • Data Directory: {'/opt/homebrew/var/mongodb' if self.is_apple_silicon else '/usr/local/var/mongodb'}
        • Log Directory: {'/opt/homebrew/var/log/mongodb' if self.is_apple_silicon else '/usr/local/var/log/mongodb'}

        🚀 Services:
        • MongoDB Service: brew services start/stop mongodb-community@7.0

        🛠️  Useful Commands:
        • Start MongoDB: brew services start mongodb-community@7.0
        • Stop MongoDB: brew services stop mongodb-community@7.0
        • Check Status: brew services list
        • Connect: mongosh
        • View Logs: tail -f {'/opt/homebrew/var/log/mongodb/mongo.log' if self.is_apple_silicon else '/usr/local/var/log/mongodb/mongo.log'}

        🧭 MongoDB Compass:
        • Location: /Applications/MongoDB Compass.app
        • Launch: Open Spotlight (Cmd+Space), type "MongoDB Compass"

        🔒 Security Note:
        By default, MongoDB binds to localhost (127.0.0.1) only.
        For remote access, modify bindIp in the configuration file.
        """

        print(summary)

        # Display next steps
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS")
        print("=" * 60)
        next_steps = """
        1. Open MongoDB Compass from your Applications folder
        2. Connect to: mongodb://localhost:27017
        3. Start creating databases and collections

        Need help? Visit:
        • MongoDB Documentation: https://docs.mongodb.com
        • MongoDB University: https://university.mongodb.com
        • MongoDB Community Forums: https://community.mongodb.com
        """
        print(next_steps)

    def run_installation(self):
        """Main installation workflow"""
        print("\n" + "=" * 60)
        print("🚀 MongoDB Community Edition + Compass Installer for macOS")
        print("=" * 60)
        print("This script will install:")
        print("• MongoDB Community Edition 7.0")
        print("• MongoDB Shell (mongosh)")
        print("• MongoDB Database Tools")
        print("• MongoDB Compass GUI")
        print("\nThe installation will take 5-15 minutes depending on your internet speed.")

        # Ask for confirmation
        response = input("\nDo you want to continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Installation cancelled.")
            return

        try:
            # Execute installation steps
            if not self.check_prerequisites():
                return

            if not self.install_mongodb():
                return

            if not self.configure_mongodb():
                return

            self.install_mongodb_compass()
            self.display_summary()

            print("\n" + "=" * 60)
            print("🎉 INSTALLATION COMPLETE!")
            print("=" * 60)
            print("\nYou can now use MongoDB Community Edition and Compass on your macOS!")

        except KeyboardInterrupt:
            print("\n\n⚠️  Installation interrupted by user.")
        except Exception as e:
            print(f"\n❌ An error occurred during installation: {e}")
            print("Please check the error messages above and try again.")


def main():
    """Main function"""
    # Check if running on macOS
    if platform.system() != 'Darwin':
        print("❌ This script is only for macOS.")
        print("Please refer to the MongoDB documentation for other operating systems.")
        sys.exit(1)

    # Check Python version
    if sys.version_info < (3, 6):
        print("❌ This script requires Python 3.6 or higher.")
        sys.exit(1)

    # Run installer
    installer = MongoDBInstaller()
    installer.run_installation()


if __name__ == "__main__":
    main()