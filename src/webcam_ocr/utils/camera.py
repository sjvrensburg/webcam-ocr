"""Camera utilities with fixed permission checking."""

import subprocess
import re
import os
import time
import grp
import pwd
from typing import Dict, List, Optional, Tuple
import cv2

def get_user_groups() -> List[int]:
    """Get list of group IDs the current user belongs to."""
    try:
        # Get username
        username = pwd.getpwuid(os.getuid()).pw_name
        
        # Get groups
        groups = [g.gr_gid for g in grp.getgrall() if username in g.gr_mem]
        
        # Add primary group
        primary_gid = pwd.getpwnam(username).pw_gid
        if primary_gid not in groups:
            groups.append(primary_gid)
            
        return groups
    except Exception as e:
        print(f"Error getting user groups: {e}")
        return []

def debug_device_permissions(device_path: str) -> Dict[str, bool]:
    """Check device permissions and access rights."""
    results = {
        'exists': False,
        'readable': False,
        'writable': False,
        'user_has_access': False,
        'groups': [],
        'device_group': None
    }
    
    try:
        results['exists'] = os.path.exists(device_path)
        if not results['exists']:
            return results
            
        # Check basic read/write permissions
        results['readable'] = os.access(device_path, os.R_OK)
        results['writable'] = os.access(device_path, os.W_OK)
        
        # Get device ownership info
        stat = os.stat(device_path)
        user_id = os.getuid()
        
        # Get user's groups
        user_groups = get_user_groups()
        results['groups'] = user_groups
        
        # Get device group
        try:
            device_group = grp.getgrgid(stat.st_gid).gr_name
            results['device_group'] = device_group
        except KeyError:
            results['device_group'] = str(stat.st_gid)
        
        # Check access
        results['user_has_access'] = (
            (stat.st_uid == user_id) or  # User owns the device
            (stat.st_gid in user_groups)  # User is in device group
        )
        
    except Exception as e:
        print(f"Error checking device permissions: {e}")
        
    return results

def get_v4l2_devices() -> List[Dict[str, str]]:
    """Get list of available V4L2 devices using v4l2-ctl."""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True,
            text=True,
            check=True
        )
        
        devices = []
        current_device = None
        
        for line in result.stdout.split('\n'):
            if not line.strip():
                continue
                
            if not line.startswith('\t'):
                current_device = line.split('(')[0].strip()
            else:
                if current_device and '/dev/video' in line:
                    path = line.strip()
                    perms = debug_device_permissions(path)
                    device_info = {
                        'name': current_device,
                        'path': path,
                        'permissions': perms
                    }
                    devices.append(device_info)
                    
        return devices
        
    except subprocess.CalledProcessError:
        return []
    except Exception as e:
        print(f"Error enumerating devices: {e}")
        return []

def test_camera_compatibility(device_path: str, verbose: bool = True) -> Tuple[bool, Optional[str]]:
    """Test if a camera device is compatible and working."""
    try:
        device_num = int(re.search(r'/dev/video(\d+)', device_path).group(1))
        
        if verbose:
            print(f"\nTesting camera: {device_path}")
            perms = debug_device_permissions(device_path)
            print(f"Device permissions:")
            print(f"  Exists: {perms['exists']}")
            print(f"  Readable: {perms['readable']}")
            print(f"  Writable: {perms['writable']}")
            print(f"  User has access: {perms['user_has_access']}")
            print(f"  Device group: {perms['device_group']}")
            print(f"  User groups: {perms['groups']}")
            
            if not perms['user_has_access']:
                print(f"\nTo fix permissions, run:")
                print(f"  sudo usermod -a -G {perms['device_group']} $USER")
                print("Then log out and back in")
        
        if not os.access(device_path, os.R_OK | os.W_OK):
            return False, "Insufficient permissions"
        
        # Try opening with direct device path first
        cap = cv2.VideoCapture(device_path)
        if not cap.isOpened():
            if verbose:
                print(f"Failed to open with path, trying index {device_num}")
            cap = cv2.VideoCapture(device_num)
            
        if not cap.isOpened():
            return False, "Failed to open camera"
            
        # Set properties that might help with virtual cameras
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Try multiple times to read a frame
        max_attempts = 3
        for attempt in range(max_attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return True, None
            if verbose:
                print(f"Read attempt {attempt + 1} failed, retrying...")
            time.sleep(0.1)
            
        cap.release()
        return False, "Failed to read frame after multiple attempts"
            
    except Exception as e:
        return False, str(e)

def get_camera_index(target_name: Optional[str] = None, verbose: bool = True) -> int:
    """Get camera index based on name or find first working camera."""
    devices = get_v4l2_devices()
    
    if verbose:
        print("\nSearching for cameras:")
        for device in devices:
            print(f"  {device['name']}")
            print(f"    Path: {device['path']}")
            perms = device['permissions']
            print(f"    Permissions: {perms}")
            if not perms['user_has_access']:
                print(f"    To fix: sudo usermod -a -G {perms['device_group']} $USER")
    
    if target_name:
        if verbose:
            print(f"\nLooking for camera matching: {target_name}")
            
        for device in devices:
            if target_name.lower() in device['name'].lower():
                device_path = device['path']
                if verbose:
                    print(f"Found matching device: {device_path}")
                    
                is_working, error = test_camera_compatibility(device_path, verbose)
                if is_working:
                    device_num = int(re.search(r'/dev/video(\d+)', device_path).group(1))
                    if verbose:
                        print(f"Successfully initialized camera {device_num}")
                    return device_num
                elif verbose:
                    print(f"Device found but not working: {error}")
    
    # Find first working camera
    if verbose:
        print("\nTrying to find first working camera...")
        
    for device in devices:
        device_path = device['path']
        is_working, _ = test_camera_compatibility(device_path, verbose=False)
        if is_working:
            device_num = int(re.search(r'/dev/video(\d+)', device_path).group(1))
            if verbose:
                print(f"Found working camera: {device_path}")
            return device_num
            
    if verbose:
        print("No working cameras found, defaulting to 0")
    return 0

def init_camera(camera_index: int, fallback_to_default: bool = True) -> Optional[cv2.VideoCapture]:
    """Initialize camera with fallback option."""
    # Try to open by index first
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        # Configure for virtual camera compatibility
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Verify we can actually read frames
        ret, _ = cap.read()
        if ret:
            return cap
        cap.release()
    
    # Try opening by device path
    device_path = f"/dev/video{camera_index}"
    if os.path.exists(device_path):
        cap = cv2.VideoCapture(device_path)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    
    if fallback_to_default and camera_index != 0:
        print(f"Failed to open camera {camera_index}, falling back to camera 0")
        return init_camera(0, fallback_to_default=False)
            
    return None
