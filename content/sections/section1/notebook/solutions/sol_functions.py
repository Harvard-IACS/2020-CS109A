def get_award_title(award_node):
    return award_node.select_one('h3').text[:-4].strip()

def get_award_year(award_node):
    return int(award_node.select_one('h3').text[-4:])