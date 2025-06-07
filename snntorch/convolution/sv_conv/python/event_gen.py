#!/usr/bin/env python3
"""
Simple SNN Event Generator - FIXED VERSION
Generates test events and exports to simple text format for testbench
"""

import random
import argparse

def generate_events(num_events, img_width=32, img_height=32, event_type="mixed"):
    """Generate test events based on specified type"""
    
    events = []
    
    if event_type == "random":
        # Generate completely random events
        for i in range(num_events):
            x = random.randint(0, img_width - 1)
            y = random.randint(0, img_height - 1)
            events.append((x, y))
    
    elif event_type == "corners":
        # Test corner cases
        corner_events = [
            (0, 0),           # Top-left corner
            (img_width-1, 0), # Top-right corner
            (0, img_height-1), # Bottom-left corner
            (img_width-1, img_height-1), # Bottom-right corner
        ]
        events.extend(corner_events[:num_events])
        
        # Fill remaining with random if needed
        while len(events) < num_events:
            x = random.randint(0, img_width - 1)
            y = random.randint(0, img_height - 1)
            events.append((x, y))
    
    elif event_type == "edges":
        # Test edge cases
        center_x, center_y = img_width // 2, img_height // 2
        edge_events = [
            (center_x, 0),          # Top edge
            (0, center_y),          # Left edge  
            (img_width-1, center_y), # Right edge
            (center_x, img_height-1), # Bottom edge
        ]
        events.extend(edge_events[:num_events])
        
        # Fill remaining with random if needed
        while len(events) < num_events:
            x = random.randint(0, img_width - 1)
            y = random.randint(0, img_height - 1)
            events.append((x, y))
    
    elif event_type == "center":
        # Focus on center region
        center_x, center_y = img_width // 2, img_height // 2
        for i in range(num_events):
            # Generate events in center region with some spread
            spread = min(5, img_width//4, img_height//4)  # Adaptive spread
            x = max(0, min(img_width-1, center_x + random.randint(-spread, spread)))
            y = max(0, min(img_height-1, center_y + random.randint(-spread, spread)))
            events.append((x, y))
    
    elif event_type == "line":
        # Generate events in a diagonal line
        for i in range(num_events):
            t = i / max(1, num_events - 1)  # Parameter from 0 to 1
            x = int(t * (img_width - 1))
            y = int(t * (img_height - 1))
            events.append((x, y))
    
    elif event_type == "pattern":
        # Generate a repeating pattern - scaled to image size
        center_x, center_y = img_width // 2, img_height // 2
        quarter_w, quarter_h = img_width // 4, img_height // 4
        
        pattern_coords = [
            (quarter_w, quarter_h),           # Top-left quadrant
            (center_x, center_y),             # Center
            (3*quarter_w, 3*quarter_h),       # Bottom-right quadrant
            (quarter_w, 3*quarter_h),         # Bottom-left quadrant
            (3*quarter_w, quarter_h),         # Top-right quadrant
            (center_x, quarter_h)             # Top-center
        ]
        for i in range(num_events):
            coord = pattern_coords[i % len(pattern_coords)]
            events.append(coord)
    
    else:  # "mixed" - default - FIXED TO USE ACTUAL IMAGE SIZE
        # Mix of different types for comprehensive testing
        center_x, center_y = img_width // 2, img_height // 2
        quarter_w, quarter_h = img_width // 4, img_height // 4
        
        events = [
            (quarter_w, quarter_h),           # Quarter region
            (0, 0),                           # Top-left corner
            (img_width-1, img_height-1),      # Bottom-right corner (FIXED)
            (center_x, 0),                    # Top edge (FIXED)
            (0, center_y),                    # Left edge (FIXED)
            (img_width-1, center_y),          # Right edge (FIXED)
            (center_x, img_height-1),         # Bottom edge (FIXED)
            (center_x, center_y),             # Center (FIXED)
            (3*quarter_w, 3*quarter_h),       # Another region (FIXED)
            (quarter_w, quarter_h + 1),       # Overlapping region (FIXED)
        ]
        
        # Fill remaining with random if needed
        while len(events) < num_events:
            x = random.randint(1, max(1, img_width - 2))  # Avoid exact edges for variety
            y = random.randint(1, max(1, img_height - 2))
            events.append((x, y))
    
    return events[:num_events]

def save_events_to_file(events, filename="test_events.txt"):
    """Save events to simple text file for testbench"""
    
    with open(filename, 'w') as f:
        f.write(f"// Generated test events: {len(events)} total\n")
        f.write(f"// Format: x,y (one event per line)\n")
        
        for i, (x, y) in enumerate(events):
            f.write(f"{x:02d},{y:02d}\n")
    
    print(f"Saved {len(events)} events to {filename}")

def save_events_to_hex(events, filename="test_events.hex"):
    """Save events to hex file for $readmemh in testbench"""
    
    with open(filename, 'w') as f:
        for x, y in events:
            # Pack coordinates into 16-bit hex value
            packed = (x << 8) | y
            f.write(f"{packed:04X}\n")
    
    print(f"Saved {len(events)} events to {filename} (hex format)")

def save_events_config(events, img_width, img_height, filename="test_events_config.sv"):
    """Save test configuration for SystemVerilog"""
    
    with open(filename, 'w') as f:
        f.write("// Auto-generated test configuration\n")
        f.write(f"localparam int NUM_TEST_EVENTS = {len(events)};\n")
        f.write(f"localparam int TEST_IMG_WIDTH = {img_width};\n")
        f.write(f"localparam int TEST_IMG_HEIGHT = {img_height};\n")
        f.write(f"localparam string TEST_EVENTS_FILE = \"test_events.hex\";\n")
        f.write(f"localparam string MEMORY_DUMP_FILE = \"memory_dumps.csv\";\n")
    
    print(f"Saved test configuration to {filename}")

def print_event_summary(events, img_width, img_height):
    """Print summary of generated events"""
    
    print(f"\n=== Event Generation Summary ===")
    print(f"Total events: {len(events)}")
    print(f"Image size: {img_width}x{img_height}")
    
    # Validate all events are within bounds
    valid_events = 0
    invalid_events = []
    for i, (x, y) in enumerate(events):
        if 0 <= x < img_width and 0 <= y < img_height:
            valid_events += 1
        else:
            invalid_events.append((i, x, y))
    
    print(f"Valid events: {valid_events}/{len(events)}")
    if invalid_events:
        print(f"❌ Invalid events found:")
        for i, x, y in invalid_events:
            print(f"  Event {i}: ({x}, {y}) - outside {img_width}x{img_height} bounds")
    else:
        print(f"✅ All events within bounds")
    
    # Analyze event distribution
    corners = sum(1 for x, y in events if (x, y) in [(0, 0), (0, img_height-1), (img_width-1, 0), (img_width-1, img_height-1)])
    edges = sum(1 for x, y in events if x == 0 or x == img_width-1 or y == 0 or y == img_height-1)
    center = sum(1 for x, y in events if img_width//4 <= x <= 3*img_width//4 and img_height//4 <= y <= 3*img_height//4)
    
    print(f"Corner events: {corners}")
    print(f"Edge events: {edges}")
    print(f"Center region events: {center}")
    
    # Show first few events
    print(f"\nFirst 10 events:")
    for i, (x, y) in enumerate(events[:10]):
        print(f"  Event {i}: ({x:2d}, {y:2d})")
    
    if len(events) > 10:
        print(f"  ... and {len(events) - 10} more events")

def main():
    parser = argparse.ArgumentParser(description='Generate SNN test events')
    parser.add_argument('-n', '--num-events', type=int, default=10, 
                       help='Number of events to generate (default: 10)')
    parser.add_argument('-t', '--type', choices=['random', 'corners', 'edges', 'center', 'line', 'pattern', 'mixed'], 
                       default='mixed', help='Type of events to generate (default: mixed)')
    parser.add_argument('-w', '--width', type=int, default=32, 
                       help='Image width (default: 32)')
    parser.add_argument('--height', type=int, default=32, 
                       help='Image height (default: 32)')
    parser.add_argument('-o', '--output', type=str, default='test_events', 
                       help='Output filename prefix (default: test_events)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    print(f"Generating {args.num_events} events of type '{args.type}'...")
    print(f"Image size: {args.width}x{args.height}")
    
    # Generate events
    events = generate_events(args.num_events, args.width, args.height, args.type)
    
    # Save in multiple formats
    save_events_to_file(events, f"{args.output}.txt")
    save_events_to_hex(events, f"{args.output}.hex")
    save_events_config(events, args.width, args.height, f"{args.output}_config.sv")
    
    # Print summary
    print_event_summary(events, args.width, args.height)
    
    print(f"\nFiles generated:")
    print(f"  - {args.output}.txt: Human-readable event list")
    print(f"  - {args.output}.hex: Hex format for $readmemh")
    print(f"  - {args.output}_config.sv: SystemVerilog configuration")
    print(f"\nReady for testbench!")

if __name__ == "__main__":
    main()