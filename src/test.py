import pandas as pd

def main():
    file = pd.read_csv('acd990.csv', header = None)
    print("done")

    output = []
    for line in file[0]:
        splitted = line.split(" ")
        output.append(splitted[0])
        print(splitted[0])
    
    df = pd.DataFrame(output)
    df.to_csv('acd990_processed.csv', index = False)

    """
    active_count = 0;
    inactive_count = 0;
    for line in file['Active']:
        if (line == 1):
            active_count += 1
        if (line == 0):
            inactive_count += 1
        
    print("Active: " + str(active_count) + " Inactive: " + str(inactive_count))
    """

if __name__ == "__main__":
    main()

    