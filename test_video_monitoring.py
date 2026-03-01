"""
Video Monitoring System - Integration Test
Tests both live monitoring and video upload functionality
"""

import requests
import time
import json
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000/api"
TOKEN = None  # Will be set after login

def test_login():
    """Test user authentication"""
    global TOKEN
    print("\n🔐 Testing Login...")
    
    response = requests.post(
        f"{API_BASE}/caretaker/login",
        json={
            "username": "test_user",  # Replace with your test credentials
            "password": "test_password"
        }
    )
    
    if response.status_code == 200:
        TOKEN = response.json().get("access_token")
        print("✅ Login successful")
        return True
    else:
        print(f"❌ Login failed: {response.text}")
        return False

def test_stats():
    """Test getting monitoring stats"""
    print("\n📊 Testing Stats Endpoint...")
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.get(f"{API_BASE}/video-monitoring/stats", headers=headers)
    
    if response.status_code == 200:
        stats = response.json()
        print(f"✅ Stats retrieved: {stats}")
        return True
    else:
        print(f"❌ Stats failed: {response.text}")
        return False

def test_live_monitoring():
    """Test live monitoring start/stop"""
    print("\n📹 Testing Live Monitoring...")
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    
    # Start monitoring
    print("  Starting live session...")
    response = requests.post(
        f"{API_BASE}/video-monitoring/start",
        headers=headers,
        json={"camera_index": 0, "sensitivity": "medium"}
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to start: {response.text}")
        return False
    
    session_data = response.json()
    session_id = session_data.get("session_id")
    stream_url = session_data.get("stream_url")
    
    print(f"✅ Session started: {session_id}")
    print(f"   Stream URL: {stream_url}")
    
    # Wait a bit
    print("  Waiting 5 seconds...")
    time.sleep(5)
    
    # Check alerts
    print("  Checking alerts...")
    response = requests.get(
        f"{API_BASE}/video-monitoring/session/{session_id}/alerts",
        headers=headers
    )
    
    if response.status_code == 200:
        alerts = response.json()
        print(f"✅ Alerts retrieved: {alerts.get('total', 0)} alerts")
    
    # Update threshold
    print("  Testing threshold update...")
    response = requests.post(
        f"{API_BASE}/video-monitoring/update-threshold/{session_id}",
        headers=headers,
        json={"threshold_seconds": 45}
    )
    
    if response.status_code == 200:
        print("✅ Threshold updated successfully")
    
    # Stop monitoring
    print("  Stopping session...")
    response = requests.post(
        f"{API_BASE}/video-monitoring/stop/{session_id}",
        headers=headers
    )
    
    if response.status_code == 200:
        print("✅ Session stopped successfully")
        return True
    else:
        print(f"❌ Failed to stop: {response.text}")
        return False

def test_video_upload():
    """Test video upload and processing"""
    print("\n📤 Testing Video Upload...")
    
    # Check if test video exists
    test_video = Path("test_video.mp4")
    if not test_video.exists():
        print("⚠️  No test video found. Skipping upload test.")
        print("   Create a test_video.mp4 file to test this feature.")
        return True
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    
    # Upload video
    print("  Uploading video...")
    with open(test_video, "rb") as f:
        files = {"file": ("test_video.mp4", f, "video/mp4")}
        response = requests.post(
            f"{API_BASE}/video-monitoring/upload-video",
            headers=headers,
            files=files
        )
    
    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return False
    
    upload_data = response.json()
    process_id = upload_data.get("process_id")
    print(f"✅ Upload successful: {process_id}")
    
    # Poll for status
    print("  Monitoring processing status...")
    max_attempts = 60  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        response = requests.get(
            f"{API_BASE}/video-monitoring/status/{process_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            status_data = response.json()
            status = status_data.get("status")
            progress = status_data.get("progress", 0)
            
            print(f"  Status: {status} - Progress: {progress}%")
            
            if status == "completed":
                print(f"✅ Processing completed!")
                print(f"   Falls detected: {status_data.get('has_falls', False)}")
                print(f"   Output file: {status_data.get('output_filename')}")
                return True
            elif status == "error":
                print(f"❌ Processing error: {status_data.get('error')}")
                return False
        
        time.sleep(5)
        attempt += 1
    
    print("❌ Processing timeout")
    return False

def test_history():
    """Test history retrieval"""
    print("\n📜 Testing History...")
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = requests.get(f"{API_BASE}/video-monitoring/history", headers=headers)
    
    if response.status_code == 200:
        history = response.json()
        print(f"✅ History retrieved: {len(history)} videos")
        if history:
            print(f"   Latest: {history[0].get('filename')}")
        return True
    else:
        print(f"❌ History failed: {response.text}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("VIDEO MONITORING SYSTEM - INTEGRATION TEST")
    print("=" * 60)
    
    results = {
        "Login": False,
        "Stats": False,
        "Live Monitoring": False,
        "Video Upload": False,
        "History": False
    }
    
    # Run tests
    if test_login():
        results["Login"] = True
        results["Stats"] = test_stats()
        results["Live Monitoring"] = test_live_monitoring()
        results["Video Upload"] = test_video_upload()
        results["History"] = test_history()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for showcase.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")

if __name__ == "__main__":
    main()
