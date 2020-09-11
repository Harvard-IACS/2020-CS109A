def get_recipients(award_node):
    """returns a list of award recipients using list comprehension.

    Args:
        award_node: HTML beautiful soup node of css_selector class <div class="by_year">
    """
    css_selector = 'h6 a'

    #Example form of list comprehension: [reassigned_value for value in list]
    recipients = [] #TODO

    
get_recipients(award_nodes[200])