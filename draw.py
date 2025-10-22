from graphviz import Digraph

dot = Digraph(comment='GenAI Salesbot Case Analysis')
dot.attr(rankdir='TB', size='8,8')

# 中心节点
dot.node('A', 'Should We Deploy a GenAI Salesbot?', shape='ellipse', style='filled', color='lightblue')

# 四个主要部分
dot.node('B1', 'Case Summary')
dot.node('B2', 'Problem to be Solved')
dot.node('B3', 'Assessment of Information')
dot.node('B4', 'Solutions & Recommendations')

dot.edges(['AB1', 'AB2', 'AB3', 'AB4'])

# 子节点示例
dot.node('C1', 'Internal Actors: CEO, CTO, etc.')
dot.node('C2', 'B2B trust & data privacy concerns')
dot.node('C3', 'Internal & External Factors')
dot.node('C4', 'Pilot deployment / Human-in-loop / Governance')

dot.edge('B1', 'C1')
dot.edge('B2', 'C2')
dot.edge('B3', 'C3')
dot.edge('B4', 'C4')

dot.render('genai_salesbot_mindmap', view=True, format='pdf')