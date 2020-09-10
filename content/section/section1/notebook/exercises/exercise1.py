def get_recipients(award_node):
    """
    This function returns a list of nobel prize award recipients from a specified html award node.
    """
    css_selector = '' #TODO
    recipients = []
    for node in award_node.select(css_selector):
        recipients.append(node.text)
    return(recipients)
    
get_recipients(award_nodes[200])