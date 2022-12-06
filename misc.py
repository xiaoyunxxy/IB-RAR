

def layer_selection(layer_feas, layer_index):
	'''
	use certain layers for computing mutual information
	'''
	if layer_index == 'all':
		return layer_feas
	else:
		layer_index = [int(i) for i in layer_index.split(',')]
	return [layer_feas[i] for i in layer_index]