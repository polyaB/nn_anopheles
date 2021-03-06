                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       from Bio import SeqIO

path_to_fasta = "E:/RNA-seq_anopheles/FASTA/AcolNg_V3.fa"
out_file = "X:/nn_anopheles/gaps_AcolNg_V3.fa"
records = list(SeqIO.parse(path_to_fasta, "fasta"))
lens_N_regions = []
coord_N_regions = []
for record in records:
    if record.name=='chrX':
        print(record.seq[47153:47253])
for record in records:
    print(record.name)
    print(len(record))
    counter=0
    gap_length=0
    start_pos = 0
    gap=False
    for nucleotide in record.seq:
        # print(nucleotide)
        [print(counter) if counter % 500000 == 0 else 0]
        if counter%500000==0:
            print(counter)
        if nucleotide=="N" or nucleotide=="n":
            # print(len_N)
            if gap_length == 0:
                start_pos=counter
                gap_length = 1
                gap = True
            else:
                gap_length+=1
        else:
            if gap:
                coord_N_regions.append((record.name, start_pos, start_pos+gap_length))
                lens_N_regions.append(gap_length)
                gap_length = 0
                gap = False
        counter+=1
    # break
print(lens_N_regions)
print(coord_N_regions)
with open(out_file, "w") as out_file:
    for coord in coord_N_regions:
        out_file.write(coord[0]+"\t"+str(coord[1])+"\t"+str(coord[2])+"\n")

# for record in records:
#     if record.name=='chrX':
#         print(record.seq[47153:47253])
