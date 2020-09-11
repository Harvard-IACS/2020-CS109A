#!/bin/bash/python3

def get_recipients(award_node):
    """returns a list of nobel prize award recipients from a specified html award node.

    Args:
        award_node: HTML beautiful soup node of css_selector class <div class="by_year">
    """
    css_selector = 'h6 a'
    return [node.text for node in award_node.select(css_selector)]