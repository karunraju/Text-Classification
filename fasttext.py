import io
import linecache

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    chunk_size = 1000
    file_cnt = 0
    line_list = []
    for line in fin:
        #tokens = line.rstrip().split(' ')
        #data[tokens[0]] = map(float, tokens[1:])

        if len(line_list) == chunk_size:
            out_file = './chunks/' +  str(file_cnt) + '.txt'
            with open(out_file, 'w') as f:
                for l in line_list:
                    f.write(l)

            line_list = []
            file_cnt += 1

        line_list.append(line)

    if len(line_list) > 0:
        out_file = './chunks/' +  str(file_cnt) + '.txt'
        with open(out_file, 'w') as f:
            for l in line_list:
                f.write(l)


#line = linecache.getline('crawl-300d-2M.vec', 1)
#print(line)

data = load_vectors('crawl-300d-2M.vec')
