
def take_n_tokens(dataloader, num_tokens):
    curr_num_tokens = 0
    for _, text in dataloader:        
        curr_num_tokens += sum(len(x.split()) for x in text)
        if curr_num_tokens < num_tokens:
            yield curr_num_tokens, text
        