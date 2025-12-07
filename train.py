import os

#0,img_000000_north.jpg,img_000000_east.jpg,img_000000_south.jpg,img_000000_west.jpg,Maine,18,43.472421,-70.719764
def extract_ground_truth(csv_path):
    ground_truth = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            print(f"Parts: {parts}")
            img_number = parts[0]
            state = parts[5]
            state_code = parts[6]
            lat = float(parts[7])
            lon = float(parts[8])
            ground_truth[img_number] = {
                'state': state,
                'state_code': state_code,
                'latitude': lat,
                'longitude': lon
            }
    return ground_truth