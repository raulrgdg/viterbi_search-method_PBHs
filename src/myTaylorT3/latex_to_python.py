def find_closer_bracket(string, i_start):
	
	#initialize number of open brackets as 1
	N_open_brckts = 1 

	#loop over the string
	for i, c in enumerate(string):
	
		#ignore all characters until startinf index
		if i<i_start: continue
		
		#track number of open brackets
		if c == r'{':
			N_open_brckts = N_open_brckts + 1
		elif c == r'}':
			N_open_brckts = N_open_brckts - 1		
		
		#If bracket has been closed, return position
		if N_open_brckts == 0:	
			return i
		
def latex_to_python(string, variables, remove_extra = None, replace_extra=None):

	#convert the variables from latex to python
	for latex_var, py_var in variables.items():
		if latex_var != py_var:
			string = string.replace(latex_var, py_var)

	#basic removing latex innecessary stuff
	latex_remove = [r'&', r'\,', r'\qquad', r'\quad', r'\cr', r'\\', r'\nonumber', r'\left.', r'\right.', r'\left', r'\right', r'\Biggl', r'\Biggr', r'\Bigg', r'\Big', r'\biggl', r'\biggr', r'\bigg', r'\bigl', r'\bigr', r'\big', ' ']
	#If more things are asked to be removed, add them
	if remove_extra is not None:
		latex_remove = latex_remove + remove_extra
	#remove the indicated strings
	for str_remove in latex_remove:
		string = string.replace(str_remove, r'')

	#basic replacing latex stuff
	latex_replace = {r'[': r'(', 
	                 r']': r')',
	                 r'\{':r'(',
	                 r'\}':r')',
	                 r'^':r'**'}
	#If more things are asked to be replaced, add them
	if replace_extra is not None:
		latex_replace.update(replace_extra)
	#replace the indicated strings
	for original_str, replace_str in latex_replace.items():
		string = string.replace(original_str, replace_str)
	
	#change the fractions by python divisions
	#loop over all fractions
	while string.find(r'\frac{') != -1:

		#find where the first fraction starts
		frac_start = string.find(r'\frac{')
		#replace this fraction by two parenthesis
		string = string.replace(r'\frac{', r'((', 1)
		
		#find the place where the bracket of that fraction was closed
		i_close = find_closer_bracket(string, frac_start)
		#at the postition where the bracket is closed, replace }{ by )/( 
		string = string[:i_close]+r')/('+string[i_close+2:]
		
		#find the place where the bracket of the denominator closes
		i_close = find_closer_bracket(string, i_close+3)
		#at the postition where the bracket is closed, replace } by ))
		string = string[:i_close]+r'))'+string[i_close+1:]
	
	#change sqrts by numpy sqrts
	while string.find(r'\sqrt{') != -1:
		#find where the first sqrt starts
		sqrt_start = string.find(r'\sqrt{')
		#replace this sqrt by numpy sqrt
		string = string.replace(r'\sqrt{', r'np.sqrt(', 1)

		#find the place where the bracket of that sqrt was closed
		i_close = find_closer_bracket(string, sqrt_start)
		#at the postition where the bracket was closed, replace } by ) 
		string = string[:i_close]+r')'+string[i_close+1:]
	
	#now replace all brackets by parenthesis
	string = string.replace(r'{', r'(')
	string = string.replace(r'}', r')')
	
	#eliminate every instace of a variable randomly surrounded by parenthesis
	for latex_var, py_var in variables.items():
		string = string.replace(r'('+py_var+r')', py_var)
		string = string.replace(r'('+py_var+r')', py_var)

	#also eliminate every instance of numbers randomly surrounded by parenthesis
	num_in_par = False
	num_sub = 0
	for i, c in enumerate(string):
		#correct i by number of paranthesis removed
		i = i - num_sub
		#look for numbers inside brackets
		if c==r'(':
			num_in_par = True
			i0 = i
		elif (c==r')') and num_in_par:
			#remove parenthesis around number
			num_sub = num_sub+2
			string = string[:i0] + string[i0+1:i] + string[i+1:]
			num_in_par = False
		elif not c.isdigit():
			num_in_par = False	
		
	#Now add multiplication signs omitted in latex
	
	#find all places involving parenthesis, where a multiplication sign is appropriatte
	#loop over the string
	num_mult = 0
	for i, c in enumerate(string):
		#correct i by number of multiplications added
		i = i + num_mult
		#look for brackets
		if c == r'(':
			#check what open brackets are preceded by and add multiplication signs accordingly
			if i>0:
				if string[i-1].isdigit(): 
					string = string[:i]+r'*'+string[i:]
					num_mult = num_mult + 1
		elif c == r')':
			#check what close brackets are followed by and add multiplication signs accordingly
			if i<(len(string)-1):
				if string[i+1].isdigit() or string[i+1].isalpha()or (string[i+1]=='('): 
					string = string[:(i+1)]+r'*'+string[(i+1):]
					num_mult = num_mult + 1
			

	#now look for all the instances of the variables
	for _, var in variables.items():
		
		#find all instances of var 
		start = 0
		while True:
			#find the start of this instance
			start = string.find(var, start)
			if start == -1: break
			
			#check what variable is followed by and add multiplication sign accordingly
			if start>0:
				if string[start-1].isdigit() or string[start-1].isalpha() or string[start-1]==')':
					string = string[:start]+r'*'+string[start:]
					start = start + 1
			
			
			#move to end of variable
			start = start+len(var)
			
			#check what variable is followed by and add multiplication sign accordingly
			if (start)<(len(string)-1):
				if string[start].isdigit() or string[start].isalpha() or string[start]=='(':
					string = string[:start]+r'*'+string[start:]
					start = start + 1						
		
	#return formated string		
	return string

