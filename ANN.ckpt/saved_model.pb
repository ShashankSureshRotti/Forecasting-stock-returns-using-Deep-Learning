��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/Final_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/Final_output/bias
�
,Adam/v/Final_output/bias/Read/ReadVariableOpReadVariableOpAdam/v/Final_output/bias*
_output_shapes
:*
dtype0
�
Adam/m/Final_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/Final_output/bias
�
,Adam/m/Final_output/bias/Read/ReadVariableOpReadVariableOpAdam/m/Final_output/bias*
_output_shapes
:*
dtype0
�
Adam/v/Final_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/v/Final_output/kernel
�
.Adam/v/Final_output/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Final_output/kernel*
_output_shapes

:*
dtype0
�
Adam/m/Final_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/m/Final_output/kernel
�
.Adam/m/Final_output/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Final_output/kernel*
_output_shapes

:*
dtype0
�
Adam/v/Output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/v/Output_layer/bias
�
,Adam/v/Output_layer/bias/Read/ReadVariableOpReadVariableOpAdam/v/Output_layer/bias*
_output_shapes
:*
dtype0
�
Adam/m/Output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/m/Output_layer/bias
�
,Adam/m/Output_layer/bias/Read/ReadVariableOpReadVariableOpAdam/m/Output_layer/bias*
_output_shapes
:*
dtype0
�
Adam/v/Output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameAdam/v/Output_layer/kernel
�
.Adam/v/Output_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Output_layer/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/Output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*+
shared_nameAdam/m/Output_layer/kernel
�
.Adam/m/Output_layer/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Output_layer/kernel*
_output_shapes
:	�*
dtype0

Adam/v/Dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/Dense_3/bias
x
'Adam/v/Dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_3/bias*
_output_shapes	
:�*
dtype0

Adam/m/Dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/Dense_3/bias
x
'Adam/m/Dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_3/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/Dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/Dense_3/kernel
�
)Adam/v/Dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_3/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/Dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/Dense_3/kernel
�
)Adam/m/Dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_3/kernel* 
_output_shapes
:
��*
dtype0

Adam/v/Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/Dense_2/bias
x
'Adam/v/Dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_2/bias*
_output_shapes	
:�*
dtype0

Adam/m/Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/Dense_2/bias
x
'Adam/m/Dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_2/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/Dense_2/kernel
�
)Adam/v/Dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/Dense_2/kernel
�
)Adam/m/Dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_2/kernel* 
_output_shapes
:
��*
dtype0

Adam/v/Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/v/Dense_1/bias
x
'Adam/v/Dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_1/bias*
_output_shapes	
:�*
dtype0

Adam/m/Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameAdam/m/Dense_1/bias
x
'Adam/m/Dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_1/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/v/Dense_1/kernel
�
)Adam/v/Dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameAdam/m/Dense_1/kernel
�
)Adam/m/Dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/v/Dense_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/v/Dense_head/bias
~
*Adam/v/Dense_head/bias/Read/ReadVariableOpReadVariableOpAdam/v/Dense_head/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/Dense_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/m/Dense_head/bias
~
*Adam/m/Dense_head/bias/Read/ReadVariableOpReadVariableOpAdam/m/Dense_head/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/Dense_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/v/Dense_head/kernel
�
,Adam/v/Dense_head/kernel/Read/ReadVariableOpReadVariableOpAdam/v/Dense_head/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/Dense_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*)
shared_nameAdam/m/Dense_head/kernel
�
,Adam/m/Dense_head/kernel/Read/ReadVariableOpReadVariableOpAdam/m/Dense_head/kernel*
_output_shapes
:	�*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
Final_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameFinal_output/bias
s
%Final_output/bias/Read/ReadVariableOpReadVariableOpFinal_output/bias*
_output_shapes
:*
dtype0
�
Final_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameFinal_output/kernel
{
'Final_output/kernel/Read/ReadVariableOpReadVariableOpFinal_output/kernel*
_output_shapes

:*
dtype0
z
Output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput_layer/bias
s
%Output_layer/bias/Read/ReadVariableOpReadVariableOpOutput_layer/bias*
_output_shapes
:*
dtype0
�
Output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*$
shared_nameOutput_layer/kernel
|
'Output_layer/kernel/Read/ReadVariableOpReadVariableOpOutput_layer/kernel*
_output_shapes
:	�*
dtype0
q
Dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameDense_3/bias
j
 Dense_3/bias/Read/ReadVariableOpReadVariableOpDense_3/bias*
_output_shapes	
:�*
dtype0
z
Dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameDense_3/kernel
s
"Dense_3/kernel/Read/ReadVariableOpReadVariableOpDense_3/kernel* 
_output_shapes
:
��*
dtype0
q
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameDense_2/bias
j
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes	
:�*
dtype0
z
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameDense_2/kernel
s
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel* 
_output_shapes
:
��*
dtype0
q
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameDense_1/bias
j
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes	
:�*
dtype0
z
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameDense_1/kernel
s
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel* 
_output_shapes
:
��*
dtype0
w
Dense_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nameDense_head/bias
p
#Dense_head/bias/Read/ReadVariableOpReadVariableOpDense_head/bias*
_output_shapes	
:�*
dtype0

Dense_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*"
shared_nameDense_head/kernel
x
%Dense_head/kernel/Read/ReadVariableOpReadVariableOpDense_head/kernel*
_output_shapes
:	�*
dtype0
�
 serving_default_Dense_head_inputPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_Dense_head_inputDense_head/kernelDense_head/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasDense_3/kernelDense_3/biasOutput_layer/kernelOutput_layer/biasFinal_output/kernelFinal_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_472392

NoOpNoOp
�b
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�b
value�bB�b B�b
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_random_generator* 
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator* 
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator* 
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias*
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses* 
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias*
Z
0
1
*2
+3
94
:5
H6
I7
W8
X9
e10
f11*
Z
0
1
*2
+3
94
:5
H6
I7
W8
X9
e10
f11*
	
g0* 
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
mtrace_0
ntrace_1
otrace_2
ptrace_3* 
6
qtrace_0
rtrace_1
strace_2
ttrace_3* 
* 
�
u
_variables
v_iterations
w_learning_rate
x_index_dict
y
_momentums
z_velocities
{_update_step_xla*

|serving_default* 

0
1*

0
1*
	
g0* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
a[
VARIABLE_VALUEDense_head/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEDense_head/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

*0
+1*

*0
+1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEDense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

90
:1*

90
:1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEDense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

H0
I1*

H0
I1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
^X
VARIABLE_VALUEDense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEDense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

W0
X1*

W0
X1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEOutput_layer/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEOutput_layer/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

e0
f1*

e0
f1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
c]
VARIABLE_VALUEFinal_output/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEFinal_output/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

�trace_0* 
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
v0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
f
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11* 
* 
* 
* 
* 
	
g0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
c]
VARIABLE_VALUEAdam/m/Dense_head/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/Dense_head/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/Dense_head/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/Dense_head/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/Dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/Dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/Dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/Dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/Dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/Dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/Dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/Dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/Dense_3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/Dense_3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/Dense_3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/Dense_3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/Output_layer/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/Output_layer/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/Output_layer/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/Output_layer/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/Final_output/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/Final_output/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/Final_output/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/Final_output/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameDense_head/kernelDense_head/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasDense_3/kernelDense_3/biasOutput_layer/kernelOutput_layer/biasFinal_output/kernelFinal_output/bias	iterationlearning_rateAdam/m/Dense_head/kernelAdam/v/Dense_head/kernelAdam/m/Dense_head/biasAdam/v/Dense_head/biasAdam/m/Dense_1/kernelAdam/v/Dense_1/kernelAdam/m/Dense_1/biasAdam/v/Dense_1/biasAdam/m/Dense_2/kernelAdam/v/Dense_2/kernelAdam/m/Dense_2/biasAdam/v/Dense_2/biasAdam/m/Dense_3/kernelAdam/v/Dense_3/kernelAdam/m/Dense_3/biasAdam/v/Dense_3/biasAdam/m/Output_layer/kernelAdam/v/Output_layer/kernelAdam/m/Output_layer/biasAdam/v/Output_layer/biasAdam/m/Final_output/kernelAdam/v/Final_output/kernelAdam/m/Final_output/biasAdam/v/Final_output/biastotal_1count_1totalcountConst*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_473415
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense_head/kernelDense_head/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/biasDense_3/kernelDense_3/biasOutput_layer/kernelOutput_layer/biasFinal_output/kernelFinal_output/bias	iterationlearning_rateAdam/m/Dense_head/kernelAdam/v/Dense_head/kernelAdam/m/Dense_head/biasAdam/v/Dense_head/biasAdam/m/Dense_1/kernelAdam/v/Dense_1/kernelAdam/m/Dense_1/biasAdam/v/Dense_1/biasAdam/m/Dense_2/kernelAdam/v/Dense_2/kernelAdam/m/Dense_2/biasAdam/v/Dense_2/biasAdam/m/Dense_3/kernelAdam/v/Dense_3/kernelAdam/m/Dense_3/biasAdam/v/Dense_3/biasAdam/m/Output_layer/kernelAdam/v/Output_layer/kernelAdam/m/Output_layer/biasAdam/v/Output_layer/biasAdam/m/Final_output/kernelAdam/v/Final_output/kernelAdam/m/Final_output/biasAdam/v/Final_output/biastotal_1count_1totalcount*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_473551ܙ
�
c
*__inference_dropout_8_layer_call_fn_473040

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_471912t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_7_layer_call_fn_472973

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_471861t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
"__inference__traced_restore_473551
file_prefix5
"assignvariableop_dense_head_kernel:	�1
"assignvariableop_1_dense_head_bias:	�5
!assignvariableop_2_dense_1_kernel:
��.
assignvariableop_3_dense_1_bias:	�5
!assignvariableop_4_dense_2_kernel:
��.
assignvariableop_5_dense_2_bias:	�5
!assignvariableop_6_dense_3_kernel:
��.
assignvariableop_7_dense_3_bias:	�9
&assignvariableop_8_output_layer_kernel:	�2
$assignvariableop_9_output_layer_bias:9
'assignvariableop_10_final_output_kernel:3
%assignvariableop_11_final_output_bias:'
assignvariableop_12_iteration:	 +
!assignvariableop_13_learning_rate: ?
,assignvariableop_14_adam_m_dense_head_kernel:	�?
,assignvariableop_15_adam_v_dense_head_kernel:	�9
*assignvariableop_16_adam_m_dense_head_bias:	�9
*assignvariableop_17_adam_v_dense_head_bias:	�=
)assignvariableop_18_adam_m_dense_1_kernel:
��=
)assignvariableop_19_adam_v_dense_1_kernel:
��6
'assignvariableop_20_adam_m_dense_1_bias:	�6
'assignvariableop_21_adam_v_dense_1_bias:	�=
)assignvariableop_22_adam_m_dense_2_kernel:
��=
)assignvariableop_23_adam_v_dense_2_kernel:
��6
'assignvariableop_24_adam_m_dense_2_bias:	�6
'assignvariableop_25_adam_v_dense_2_bias:	�=
)assignvariableop_26_adam_m_dense_3_kernel:
��=
)assignvariableop_27_adam_v_dense_3_kernel:
��6
'assignvariableop_28_adam_m_dense_3_bias:	�6
'assignvariableop_29_adam_v_dense_3_bias:	�A
.assignvariableop_30_adam_m_output_layer_kernel:	�A
.assignvariableop_31_adam_v_output_layer_kernel:	�:
,assignvariableop_32_adam_m_output_layer_bias::
,assignvariableop_33_adam_v_output_layer_bias:@
.assignvariableop_34_adam_m_final_output_kernel:@
.assignvariableop_35_adam_v_final_output_kernel::
,assignvariableop_36_adam_m_final_output_bias::
,assignvariableop_37_adam_v_final_output_bias:%
assignvariableop_38_total_1: %
assignvariableop_39_count_1: #
assignvariableop_40_total: #
assignvariableop_41_count: 
identity_43��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp"assignvariableop_dense_head_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_head_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_output_layer_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_output_layer_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_final_output_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_final_output_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_iterationIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_adam_m_dense_head_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_v_dense_head_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_m_dense_head_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_v_dense_head_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_m_dense_1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_v_dense_1_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_m_dense_1_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_v_dense_1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_dense_2_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_dense_2_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_m_dense_2_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_v_dense_2_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_m_dense_3_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_v_dense_3_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_m_dense_3_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_v_dense_3_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_adam_m_output_layer_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_v_output_layer_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_m_output_layer_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_v_output_layer_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_m_final_output_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_v_final_output_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_m_final_output_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_v_final_output_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_1Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpassignvariableop_40_totalIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpassignvariableop_41_countIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
C__inference_Dense_2_layer_call_and_return_conditional_losses_472968

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
#__inference__update_step_xla_442304
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: *
	_noinline(:I E

_output_shapes
:	�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
-__inference_Final_output_layer_call_fn_473121

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Final_output_layer_call_and_return_conditional_losses_471968o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_471991

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_472861

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_473062

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_472928

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
H__inference_Output_layer_layer_call_and_return_conditional_losses_471944

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_Output_layer_layer_call_fn_473071

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_471944s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�5
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_472042
dense_head_input$
dense_head_471982:	� 
dense_head_471984:	�"
dense_1_471993:
��
dense_1_471995:	�"
dense_2_472004:
��
dense_2_472006:	�"
dense_3_472015:
��
dense_3_472017:	�&
output_layer_472026:	�!
output_layer_472028:%
final_output_472032:!
final_output_472034:
identity��Dense_1/StatefulPartitionedCall�Dense_2/StatefulPartitionedCall�Dense_3/StatefulPartitionedCall�"Dense_head/StatefulPartitionedCall�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�$Final_output/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�
"Dense_head/StatefulPartitionedCallStatefulPartitionedCalldense_head_inputdense_head_471982dense_head_471984*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_Dense_head_layer_call_and_return_conditional_losses_471741�
dropout_5/PartitionedCallPartitionedCall+Dense_head/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_471991�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_1_471993dense_1_471995*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_1_layer_call_and_return_conditional_losses_471792�
dropout_6/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_472002�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_2_472004dense_2_472006*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_2_layer_call_and_return_conditional_losses_471843�
dropout_7/PartitionedCallPartitionedCall(Dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_472013�
Dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_3_472015dense_3_472017*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_3_layer_call_and_return_conditional_losses_471894�
dropout_8/PartitionedCallPartitionedCall(Dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_472024�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0output_layer_472026output_layer_472028*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_471944�
flatten/PartitionedCallPartitionedCall-Output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_471956�
$Final_output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0final_output_472032final_output_472034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Final_output_layer_call_and_return_conditional_losses_471968�
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_head_471982*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Final_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^Dense_3/StatefulPartitionedCall#^Dense_head/StatefulPartitionedCall4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp%^Final_output/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
Dense_3/StatefulPartitionedCallDense_3/StatefulPartitionedCall2H
"Dense_head/StatefulPartitionedCall"Dense_head/StatefulPartitionedCall2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp2L
$Final_output/StatefulPartitionedCall$Final_output/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:] Y
+
_output_shapes
:���������
*
_user_specified_nameDense_head_input
�
D
(__inference_flatten_layer_call_fn_473106

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_471956`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_472856

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_472990

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_8_layer_call_fn_473045

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_472024e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
Q
#__inference__update_step_xla_442334
gradient
variable:
��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: *
	_noinline(:R N
(
_output_shapes
:����������
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�5
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_472160

inputs$
dense_head_472120:	� 
dense_head_472122:	�"
dense_1_472126:
��
dense_1_472128:	�"
dense_2_472132:
��
dense_2_472134:	�"
dense_3_472138:
��
dense_3_472140:	�&
output_layer_472144:	�!
output_layer_472146:%
final_output_472150:!
final_output_472152:
identity��Dense_1/StatefulPartitionedCall�Dense_2/StatefulPartitionedCall�Dense_3/StatefulPartitionedCall�"Dense_head/StatefulPartitionedCall�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�$Final_output/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�
"Dense_head/StatefulPartitionedCallStatefulPartitionedCallinputsdense_head_472120dense_head_472122*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_Dense_head_layer_call_and_return_conditional_losses_471741�
dropout_5/PartitionedCallPartitionedCall+Dense_head/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_471991�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0dense_1_472126dense_1_472128*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_1_layer_call_and_return_conditional_losses_471792�
dropout_6/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_472002�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_6/PartitionedCall:output:0dense_2_472132dense_2_472134*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_2_layer_call_and_return_conditional_losses_471843�
dropout_7/PartitionedCallPartitionedCall(Dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_472013�
Dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0dense_3_472138dense_3_472140*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_3_layer_call_and_return_conditional_losses_471894�
dropout_8/PartitionedCallPartitionedCall(Dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_472024�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0output_layer_472144output_layer_472146*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_471944�
flatten/PartitionedCallPartitionedCall-Output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_471956�
$Final_output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0final_output_472150final_output_472152*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Final_output_layer_call_and_return_conditional_losses_471968�
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_head_472120*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Final_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^Dense_3/StatefulPartitionedCall#^Dense_head/StatefulPartitionedCall4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp%^Final_output/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
Dense_3/StatefulPartitionedCallDense_3/StatefulPartitionedCall2H
"Dense_head/StatefulPartitionedCall"Dense_head/StatefulPartitionedCall2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp2L
$Final_output/StatefulPartitionedCall$Final_output/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
E__inference_dropout_6_layer_call_and_return_conditional_losses_472002

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_6_layer_call_and_return_conditional_losses_472923

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�

H__inference_sequential_1_layer_call_and_return_conditional_losses_472636

inputs?
,dense_head_tensordot_readvariableop_resource:	�9
*dense_head_biasadd_readvariableop_resource:	�=
)dense_1_tensordot_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�=
)dense_2_tensordot_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�=
)dense_3_tensordot_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�A
.output_layer_tensordot_readvariableop_resource:	�:
,output_layer_biasadd_readvariableop_resource:=
+final_output_matmul_readvariableop_resource::
,final_output_biasadd_readvariableop_resource:
identity��Dense_1/BiasAdd/ReadVariableOp� Dense_1/Tensordot/ReadVariableOp�Dense_2/BiasAdd/ReadVariableOp� Dense_2/Tensordot/ReadVariableOp�Dense_3/BiasAdd/ReadVariableOp� Dense_3/Tensordot/ReadVariableOp�!Dense_head/BiasAdd/ReadVariableOp�#Dense_head/Tensordot/ReadVariableOp�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�#Final_output/BiasAdd/ReadVariableOp�"Final_output/MatMul/ReadVariableOp�#Output_layer/BiasAdd/ReadVariableOp�%Output_layer/Tensordot/ReadVariableOp�
#Dense_head/Tensordot/ReadVariableOpReadVariableOp,dense_head_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0c
Dense_head/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:j
Dense_head/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ^
Dense_head/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��d
"Dense_head/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/GatherV2GatherV2#Dense_head/Tensordot/Shape:output:0"Dense_head/Tensordot/free:output:0+Dense_head/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:f
$Dense_head/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/GatherV2_1GatherV2#Dense_head/Tensordot/Shape:output:0"Dense_head/Tensordot/axes:output:0-Dense_head/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
Dense_head/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_head/Tensordot/ProdProd&Dense_head/Tensordot/GatherV2:output:0#Dense_head/Tensordot/Const:output:0*
T0*
_output_shapes
: f
Dense_head/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_head/Tensordot/Prod_1Prod(Dense_head/Tensordot/GatherV2_1:output:0%Dense_head/Tensordot/Const_1:output:0*
T0*
_output_shapes
: b
 Dense_head/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/concatConcatV2"Dense_head/Tensordot/free:output:0"Dense_head/Tensordot/axes:output:0)Dense_head/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_head/Tensordot/stackPack"Dense_head/Tensordot/Prod:output:0$Dense_head/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_head/Tensordot/transpose	Transposeinputs$Dense_head/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
Dense_head/Tensordot/ReshapeReshape"Dense_head/Tensordot/transpose:y:0#Dense_head/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_head/Tensordot/MatMulMatMul%Dense_head/Tensordot/Reshape:output:0+Dense_head/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
Dense_head/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�d
"Dense_head/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/concat_1ConcatV2&Dense_head/Tensordot/GatherV2:output:0%Dense_head/Tensordot/Const_2:output:0+Dense_head/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_head/TensordotReshape%Dense_head/Tensordot/MatMul:product:0&Dense_head/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
!Dense_head/BiasAdd/ReadVariableOpReadVariableOp*dense_head_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_head/BiasAddBiasAddDense_head/Tensordot:output:0)Dense_head/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������k
Dense_head/ReluReluDense_head/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_5/dropout/MulMulDense_head/Relu:activations:0 dropout_5/dropout/Const:output:0*
T0*,
_output_shapes
:����������r
dropout_5/dropout/ShapeShapeDense_head/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0e
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������^
dropout_5/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_5/dropout/SelectV2SelectV2"dropout_5/dropout/GreaterEqual:z:0dropout_5/dropout/Mul:z:0"dropout_5/dropout/Const_1:output:0*
T0*,
_output_shapes
:�����������
 Dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
Dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
Dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
Dense_1/Tensordot/ShapeShape#dropout_5/dropout/SelectV2:output:0*
T0*
_output_shapes
::��a
Dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/GatherV2GatherV2 Dense_1/Tensordot/Shape:output:0Dense_1/Tensordot/free:output:0(Dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!Dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/GatherV2_1GatherV2 Dense_1/Tensordot/Shape:output:0Dense_1/Tensordot/axes:output:0*Dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
Dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_1/Tensordot/ProdProd#Dense_1/Tensordot/GatherV2:output:0 Dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
Dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_1/Tensordot/Prod_1Prod%Dense_1/Tensordot/GatherV2_1:output:0"Dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
Dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/concatConcatV2Dense_1/Tensordot/free:output:0Dense_1/Tensordot/axes:output:0&Dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_1/Tensordot/stackPackDense_1/Tensordot/Prod:output:0!Dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_1/Tensordot/transpose	Transpose#dropout_5/dropout/SelectV2:output:0!Dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Dense_1/Tensordot/ReshapeReshapeDense_1/Tensordot/transpose:y:0 Dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_1/Tensordot/MatMulMatMul"Dense_1/Tensordot/Reshape:output:0(Dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
Dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
Dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/concat_1ConcatV2#Dense_1/Tensordot/GatherV2:output:0"Dense_1/Tensordot/Const_2:output:0(Dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_1/TensordotReshape"Dense_1/Tensordot/MatMul:product:0#Dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_1/BiasAddBiasAddDense_1/Tensordot:output:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_6/dropout/MulMulDense_1/Relu:activations:0 dropout_6/dropout/Const:output:0*
T0*,
_output_shapes
:����������o
dropout_6/dropout/ShapeShapeDense_1/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0e
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������^
dropout_6/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_6/dropout/SelectV2SelectV2"dropout_6/dropout/GreaterEqual:z:0dropout_6/dropout/Mul:z:0"dropout_6/dropout/Const_1:output:0*
T0*,
_output_shapes
:�����������
 Dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
Dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
Dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
Dense_2/Tensordot/ShapeShape#dropout_6/dropout/SelectV2:output:0*
T0*
_output_shapes
::��a
Dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/GatherV2GatherV2 Dense_2/Tensordot/Shape:output:0Dense_2/Tensordot/free:output:0(Dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!Dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/GatherV2_1GatherV2 Dense_2/Tensordot/Shape:output:0Dense_2/Tensordot/axes:output:0*Dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
Dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_2/Tensordot/ProdProd#Dense_2/Tensordot/GatherV2:output:0 Dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
Dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_2/Tensordot/Prod_1Prod%Dense_2/Tensordot/GatherV2_1:output:0"Dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
Dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/concatConcatV2Dense_2/Tensordot/free:output:0Dense_2/Tensordot/axes:output:0&Dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_2/Tensordot/stackPackDense_2/Tensordot/Prod:output:0!Dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_2/Tensordot/transpose	Transpose#dropout_6/dropout/SelectV2:output:0!Dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Dense_2/Tensordot/ReshapeReshapeDense_2/Tensordot/transpose:y:0 Dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_2/Tensordot/MatMulMatMul"Dense_2/Tensordot/Reshape:output:0(Dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
Dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
Dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/concat_1ConcatV2#Dense_2/Tensordot/GatherV2:output:0"Dense_2/Tensordot/Const_2:output:0(Dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_2/TensordotReshape"Dense_2/Tensordot/MatMul:product:0#Dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_2/BiasAddBiasAddDense_2/Tensordot:output:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_7/dropout/MulMulDense_2/Relu:activations:0 dropout_7/dropout/Const:output:0*
T0*,
_output_shapes
:����������o
dropout_7/dropout/ShapeShapeDense_2/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0e
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������^
dropout_7/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_7/dropout/SelectV2SelectV2"dropout_7/dropout/GreaterEqual:z:0dropout_7/dropout/Mul:z:0"dropout_7/dropout/Const_1:output:0*
T0*,
_output_shapes
:�����������
 Dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
Dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
Dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       x
Dense_3/Tensordot/ShapeShape#dropout_7/dropout/SelectV2:output:0*
T0*
_output_shapes
::��a
Dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/GatherV2GatherV2 Dense_3/Tensordot/Shape:output:0Dense_3/Tensordot/free:output:0(Dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!Dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/GatherV2_1GatherV2 Dense_3/Tensordot/Shape:output:0Dense_3/Tensordot/axes:output:0*Dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
Dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_3/Tensordot/ProdProd#Dense_3/Tensordot/GatherV2:output:0 Dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
Dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_3/Tensordot/Prod_1Prod%Dense_3/Tensordot/GatherV2_1:output:0"Dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
Dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/concatConcatV2Dense_3/Tensordot/free:output:0Dense_3/Tensordot/axes:output:0&Dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_3/Tensordot/stackPackDense_3/Tensordot/Prod:output:0!Dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_3/Tensordot/transpose	Transpose#dropout_7/dropout/SelectV2:output:0!Dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Dense_3/Tensordot/ReshapeReshapeDense_3/Tensordot/transpose:y:0 Dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_3/Tensordot/MatMulMatMul"Dense_3/Tensordot/Reshape:output:0(Dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
Dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
Dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/concat_1ConcatV2#Dense_3/Tensordot/GatherV2:output:0"Dense_3/Tensordot/Const_2:output:0(Dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_3/TensordotReshape"Dense_3/Tensordot/MatMul:product:0#Dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
Dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_3/BiasAddBiasAddDense_3/Tensordot:output:0&Dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
Dense_3/ReluReluDense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
dropout_8/dropout/MulMulDense_3/Relu:activations:0 dropout_8/dropout/Const:output:0*
T0*,
_output_shapes
:����������o
dropout_8/dropout/ShapeShapeDense_3/Relu:activations:0*
T0*
_output_shapes
::���
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������^
dropout_8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout_8/dropout/SelectV2SelectV2"dropout_8/dropout/GreaterEqual:z:0dropout_8/dropout/Mul:z:0"dropout_8/dropout/Const_1:output:0*
T0*,
_output_shapes
:�����������
%Output_layer/Tensordot/ReadVariableOpReadVariableOp.output_layer_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0e
Output_layer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
Output_layer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
Output_layer/Tensordot/ShapeShape#dropout_8/dropout/SelectV2:output:0*
T0*
_output_shapes
::��f
$Output_layer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_layer/Tensordot/GatherV2GatherV2%Output_layer/Tensordot/Shape:output:0$Output_layer/Tensordot/free:output:0-Output_layer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
&Output_layer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Output_layer/Tensordot/GatherV2_1GatherV2%Output_layer/Tensordot/Shape:output:0$Output_layer/Tensordot/axes:output:0/Output_layer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:f
Output_layer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Output_layer/Tensordot/ProdProd(Output_layer/Tensordot/GatherV2:output:0%Output_layer/Tensordot/Const:output:0*
T0*
_output_shapes
: h
Output_layer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Output_layer/Tensordot/Prod_1Prod*Output_layer/Tensordot/GatherV2_1:output:0'Output_layer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: d
"Output_layer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_layer/Tensordot/concatConcatV2$Output_layer/Tensordot/free:output:0$Output_layer/Tensordot/axes:output:0+Output_layer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Output_layer/Tensordot/stackPack$Output_layer/Tensordot/Prod:output:0&Output_layer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
 Output_layer/Tensordot/transpose	Transpose#dropout_8/dropout/SelectV2:output:0&Output_layer/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Output_layer/Tensordot/ReshapeReshape$Output_layer/Tensordot/transpose:y:0%Output_layer/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Output_layer/Tensordot/MatMulMatMul'Output_layer/Tensordot/Reshape:output:0-Output_layer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
Output_layer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:f
$Output_layer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_layer/Tensordot/concat_1ConcatV2(Output_layer/Tensordot/GatherV2:output:0'Output_layer/Tensordot/Const_2:output:0-Output_layer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Output_layer/TensordotReshape'Output_layer/Tensordot/MatMul:product:0(Output_layer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output_layer/BiasAddBiasAddOutput_layer/Tensordot:output:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeOutput_layer/BiasAdd:output:0flatten/Const:output:0*
T0*'
_output_shapes
:����������
"Final_output/MatMul/ReadVariableOpReadVariableOp+final_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Final_output/MatMulMatMulflatten/Reshape:output:0*Final_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#Final_output/BiasAdd/ReadVariableOpReadVariableOp,final_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Final_output/BiasAddBiasAddFinal_output/MatMul:product:0+Final_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,dense_head_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: l
IdentityIdentityFinal_output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Dense_1/BiasAdd/ReadVariableOp!^Dense_1/Tensordot/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp!^Dense_2/Tensordot/ReadVariableOp^Dense_3/BiasAdd/ReadVariableOp!^Dense_3/Tensordot/ReadVariableOp"^Dense_head/BiasAdd/ReadVariableOp$^Dense_head/Tensordot/ReadVariableOp4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp$^Final_output/BiasAdd/ReadVariableOp#^Final_output/MatMul/ReadVariableOp$^Output_layer/BiasAdd/ReadVariableOp&^Output_layer/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2D
 Dense_1/Tensordot/ReadVariableOp Dense_1/Tensordot/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2D
 Dense_2/Tensordot/ReadVariableOp Dense_2/Tensordot/ReadVariableOp2@
Dense_3/BiasAdd/ReadVariableOpDense_3/BiasAdd/ReadVariableOp2D
 Dense_3/Tensordot/ReadVariableOp Dense_3/Tensordot/ReadVariableOp2F
!Dense_head/BiasAdd/ReadVariableOp!Dense_head/BiasAdd/ReadVariableOp2J
#Dense_head/Tensordot/ReadVariableOp#Dense_head/Tensordot/ReadVariableOp2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp2J
#Final_output/BiasAdd/ReadVariableOp#Final_output/BiasAdd/ReadVariableOp2H
"Final_output/MatMul/ReadVariableOp"Final_output/MatMul/ReadVariableOp2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2N
%Output_layer/Tensordot/ReadVariableOp%Output_layer/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
*__inference_dropout_6_layer_call_fn_472906

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_471810t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�;
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_471979
dense_head_input$
dense_head_471742:	� 
dense_head_471744:	�"
dense_1_471793:
��
dense_1_471795:	�"
dense_2_471844:
��
dense_2_471846:	�"
dense_3_471895:
��
dense_3_471897:	�&
output_layer_471945:	�!
output_layer_471947:%
final_output_471969:!
final_output_471971:
identity��Dense_1/StatefulPartitionedCall�Dense_2/StatefulPartitionedCall�Dense_3/StatefulPartitionedCall�"Dense_head/StatefulPartitionedCall�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�$Final_output/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�!dropout_7/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�
"Dense_head/StatefulPartitionedCallStatefulPartitionedCalldense_head_inputdense_head_471742dense_head_471744*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_Dense_head_layer_call_and_return_conditional_losses_471741�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall+Dense_head/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_471759�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_1_471793dense_1_471795*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_1_layer_call_and_return_conditional_losses_471792�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_471810�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_2_471844dense_2_471846*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_2_layer_call_and_return_conditional_losses_471843�
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_471861�
Dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_3_471895dense_3_471897*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_3_layer_call_and_return_conditional_losses_471894�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(Dense_3/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_471912�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0output_layer_471945output_layer_471947*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_471944�
flatten/PartitionedCallPartitionedCall-Output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_471956�
$Final_output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0final_output_471969final_output_471971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Final_output_layer_call_and_return_conditional_losses_471968�
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_head_471742*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Final_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^Dense_3/StatefulPartitionedCall#^Dense_head/StatefulPartitionedCall4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp%^Final_output/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
Dense_3/StatefulPartitionedCallDense_3/StatefulPartitionedCall2H
"Dense_head/StatefulPartitionedCall"Dense_head/StatefulPartitionedCall2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp2L
$Final_output/StatefulPartitionedCall$Final_output/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:] Y
+
_output_shapes
:���������
*
_user_specified_nameDense_head_input
Ѯ
�&
__inference__traced_save_473415
file_prefix;
(read_disablecopyonread_dense_head_kernel:	�7
(read_1_disablecopyonread_dense_head_bias:	�;
'read_2_disablecopyonread_dense_1_kernel:
��4
%read_3_disablecopyonread_dense_1_bias:	�;
'read_4_disablecopyonread_dense_2_kernel:
��4
%read_5_disablecopyonread_dense_2_bias:	�;
'read_6_disablecopyonread_dense_3_kernel:
��4
%read_7_disablecopyonread_dense_3_bias:	�?
,read_8_disablecopyonread_output_layer_kernel:	�8
*read_9_disablecopyonread_output_layer_bias:?
-read_10_disablecopyonread_final_output_kernel:9
+read_11_disablecopyonread_final_output_bias:-
#read_12_disablecopyonread_iteration:	 1
'read_13_disablecopyonread_learning_rate: E
2read_14_disablecopyonread_adam_m_dense_head_kernel:	�E
2read_15_disablecopyonread_adam_v_dense_head_kernel:	�?
0read_16_disablecopyonread_adam_m_dense_head_bias:	�?
0read_17_disablecopyonread_adam_v_dense_head_bias:	�C
/read_18_disablecopyonread_adam_m_dense_1_kernel:
��C
/read_19_disablecopyonread_adam_v_dense_1_kernel:
��<
-read_20_disablecopyonread_adam_m_dense_1_bias:	�<
-read_21_disablecopyonread_adam_v_dense_1_bias:	�C
/read_22_disablecopyonread_adam_m_dense_2_kernel:
��C
/read_23_disablecopyonread_adam_v_dense_2_kernel:
��<
-read_24_disablecopyonread_adam_m_dense_2_bias:	�<
-read_25_disablecopyonread_adam_v_dense_2_bias:	�C
/read_26_disablecopyonread_adam_m_dense_3_kernel:
��C
/read_27_disablecopyonread_adam_v_dense_3_kernel:
��<
-read_28_disablecopyonread_adam_m_dense_3_bias:	�<
-read_29_disablecopyonread_adam_v_dense_3_bias:	�G
4read_30_disablecopyonread_adam_m_output_layer_kernel:	�G
4read_31_disablecopyonread_adam_v_output_layer_kernel:	�@
2read_32_disablecopyonread_adam_m_output_layer_bias:@
2read_33_disablecopyonread_adam_v_output_layer_bias:F
4read_34_disablecopyonread_adam_m_final_output_kernel:F
4read_35_disablecopyonread_adam_v_final_output_kernel:@
2read_36_disablecopyonread_adam_m_final_output_bias:@
2read_37_disablecopyonread_adam_v_final_output_bias:+
!read_38_disablecopyonread_total_1: +
!read_39_disablecopyonread_count_1: )
read_40_disablecopyonread_total: )
read_41_disablecopyonread_count: 
savev2_const
identity_85��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: z
Read/DisableCopyOnReadDisableCopyOnRead(read_disablecopyonread_dense_head_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp(read_disablecopyonread_dense_head_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	�|
Read_1/DisableCopyOnReadDisableCopyOnRead(read_1_disablecopyonread_dense_head_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp(read_1_disablecopyonread_dense_head_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_output_layer_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	�~
Read_9/DisableCopyOnReadDisableCopyOnRead*read_9_disablecopyonread_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp*read_9_disablecopyonread_output_layer_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_10/DisableCopyOnReadDisableCopyOnRead-read_10_disablecopyonread_final_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp-read_10_disablecopyonread_final_output_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_11/DisableCopyOnReadDisableCopyOnRead+read_11_disablecopyonread_final_output_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp+read_11_disablecopyonread_final_output_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_12/DisableCopyOnReadDisableCopyOnRead#read_12_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp#read_12_disablecopyonread_iteration^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_learning_rate^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_14/DisableCopyOnReadDisableCopyOnRead2read_14_disablecopyonread_adam_m_dense_head_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp2read_14_disablecopyonread_adam_m_dense_head_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_15/DisableCopyOnReadDisableCopyOnRead2read_15_disablecopyonread_adam_v_dense_head_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp2read_15_disablecopyonread_adam_v_dense_head_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_16/DisableCopyOnReadDisableCopyOnRead0read_16_disablecopyonread_adam_m_dense_head_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp0read_16_disablecopyonread_adam_m_dense_head_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnRead0read_17_disablecopyonread_adam_v_dense_head_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp0read_17_disablecopyonread_adam_v_dense_head_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_adam_m_dense_1_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_19/DisableCopyOnReadDisableCopyOnRead/read_19_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp/read_19_disablecopyonread_adam_v_dense_1_kernel^Read_19/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_adam_m_dense_1_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_21/DisableCopyOnReadDisableCopyOnRead-read_21_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp-read_21_disablecopyonread_adam_v_dense_1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_m_dense_2_kernel^Read_22/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_23/DisableCopyOnReadDisableCopyOnRead/read_23_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp/read_23_disablecopyonread_adam_v_dense_2_kernel^Read_23/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_adam_m_dense_2_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_25/DisableCopyOnReadDisableCopyOnRead-read_25_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp-read_25_disablecopyonread_adam_v_dense_2_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead/read_26_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp/read_26_disablecopyonread_adam_m_dense_3_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_adam_v_dense_3_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_28/DisableCopyOnReadDisableCopyOnRead-read_28_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp-read_28_disablecopyonread_adam_m_dense_3_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_29/DisableCopyOnReadDisableCopyOnRead-read_29_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp-read_29_disablecopyonread_adam_v_dense_3_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_30/DisableCopyOnReadDisableCopyOnRead4read_30_disablecopyonread_adam_m_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp4read_30_disablecopyonread_adam_m_output_layer_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_31/DisableCopyOnReadDisableCopyOnRead4read_31_disablecopyonread_adam_v_output_layer_kernel"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp4read_31_disablecopyonread_adam_v_output_layer_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_32/DisableCopyOnReadDisableCopyOnRead2read_32_disablecopyonread_adam_m_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp2read_32_disablecopyonread_adam_m_output_layer_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_33/DisableCopyOnReadDisableCopyOnRead2read_33_disablecopyonread_adam_v_output_layer_bias"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp2read_33_disablecopyonread_adam_v_output_layer_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_34/DisableCopyOnReadDisableCopyOnRead4read_34_disablecopyonread_adam_m_final_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp4read_34_disablecopyonread_adam_m_final_output_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_35/DisableCopyOnReadDisableCopyOnRead4read_35_disablecopyonread_adam_v_final_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp4read_35_disablecopyonread_adam_v_final_output_kernel^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:�
Read_36/DisableCopyOnReadDisableCopyOnRead2read_36_disablecopyonread_adam_m_final_output_bias"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp2read_36_disablecopyonread_adam_m_final_output_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_37/DisableCopyOnReadDisableCopyOnRead2read_37_disablecopyonread_adam_v_final_output_bias"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp2read_37_disablecopyonread_adam_v_final_output_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_38/DisableCopyOnReadDisableCopyOnRead!read_38_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp!read_38_disablecopyonread_total_1^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_39/DisableCopyOnReadDisableCopyOnRead!read_39_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp!read_39_disablecopyonread_count_1^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_40/DisableCopyOnReadDisableCopyOnReadread_40_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpread_40_disablecopyonread_total^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_41/DisableCopyOnReadDisableCopyOnReadread_41_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpread_41_disablecopyonread_count^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*�
value�B�+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *9
dtypes/
-2+	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_84Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_85IdentityIdentity_84:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_85Identity_85:output:0*k
_input_shapesZ
X: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:+

_output_shapes
: 
�
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_472995

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_472454

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_472160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
H__inference_Final_output_layer_call_and_return_conditional_losses_471968

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_Dense_3_layer_call_and_return_conditional_losses_473035

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_Dense_1_layer_call_and_return_conditional_losses_472901

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_7_layer_call_and_return_conditional_losses_472013

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_Dense_2_layer_call_fn_472937

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_2_layer_call_and_return_conditional_losses_471843t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
#__inference__update_step_xla_442344
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: *
	_noinline(:Q M
'
_output_shapes
:���������
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

�
$__inference_signature_wrapper_472392
dense_head_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_head_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_471702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:���������
*
_user_specified_nameDense_head_input
�
�
-__inference_sequential_1_layer_call_fn_472425

inputs
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_472088o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_Dense_1_layer_call_fn_472870

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_1_layer_call_and_return_conditional_losses_471792t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
__inference_loss_fn_0_473140O
<dense_head_kernel_regularizer_l2loss_readvariableop_resource:	�
identity��3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<dense_head_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%Dense_head/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: |
NoOpNoOp4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_473112

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�;
�
H__inference_sequential_1_layer_call_and_return_conditional_losses_472088

inputs$
dense_head_472048:	� 
dense_head_472050:	�"
dense_1_472054:
��
dense_1_472056:	�"
dense_2_472060:
��
dense_2_472062:	�"
dense_3_472066:
��
dense_3_472068:	�&
output_layer_472072:	�!
output_layer_472074:%
final_output_472078:!
final_output_472080:
identity��Dense_1/StatefulPartitionedCall�Dense_2/StatefulPartitionedCall�Dense_3/StatefulPartitionedCall�"Dense_head/StatefulPartitionedCall�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�$Final_output/StatefulPartitionedCall�$Output_layer/StatefulPartitionedCall�!dropout_5/StatefulPartitionedCall�!dropout_6/StatefulPartitionedCall�!dropout_7/StatefulPartitionedCall�!dropout_8/StatefulPartitionedCall�
"Dense_head/StatefulPartitionedCallStatefulPartitionedCallinputsdense_head_472048dense_head_472050*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_Dense_head_layer_call_and_return_conditional_losses_471741�
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall+Dense_head/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_471759�
Dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0dense_1_472054dense_1_472056*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_1_layer_call_and_return_conditional_losses_471792�
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_471810�
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_6/StatefulPartitionedCall:output:0dense_2_472060dense_2_472062*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_2_layer_call_and_return_conditional_losses_471843�
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(Dense_2/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_471861�
Dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0dense_3_472066dense_3_472068*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_3_layer_call_and_return_conditional_losses_471894�
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(Dense_3/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_471912�
$Output_layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0output_layer_472072output_layer_472074*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Output_layer_layer_call_and_return_conditional_losses_471944�
flatten/PartitionedCallPartitionedCall-Output_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_471956�
$Final_output/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0final_output_472078final_output_472080*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_Final_output_layer_call_and_return_conditional_losses_471968�
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_head_472048*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: |
IdentityIdentity-Final_output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^Dense_3/StatefulPartitionedCall#^Dense_head/StatefulPartitionedCall4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp%^Final_output/StatefulPartitionedCall%^Output_layer/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
Dense_3/StatefulPartitionedCallDense_3/StatefulPartitionedCall2H
"Dense_head/StatefulPartitionedCall"Dense_head/StatefulPartitionedCall2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp2L
$Final_output/StatefulPartitionedCall$Final_output/StatefulPartitionedCall2L
$Output_layer/StatefulPartitionedCall$Output_layer/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_Dense_1_layer_call_and_return_conditional_losses_471792

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_7_layer_call_fn_472978

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_7_layer_call_and_return_conditional_losses_472013e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
*__inference_dropout_5_layer_call_fn_472839

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_471759t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_5_layer_call_fn_472844

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_471991e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_Dense_2_layer_call_and_return_conditional_losses_471843

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�
F__inference_Dense_head_layer_call_and_return_conditional_losses_472834

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:�����������
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_471912

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_442329
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
Q
#__inference__update_step_xla_442314
gradient
variable:
��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: *
	_noinline(:R N
(
_output_shapes
:����������
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�#
�
F__inference_Dense_head_layer_call_and_return_conditional_losses_471741

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:�����������
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:�����������
NoOpNoOp^BiasAdd/ReadVariableOp4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
Q
#__inference__update_step_xla_442324
gradient
variable:
��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: *
	_noinline(:R N
(
_output_shapes
:����������
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
L
#__inference__update_step_xla_442309
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_471759

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_442339
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_471956

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_442319
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:E A

_output_shapes	
:�
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
K
#__inference__update_step_xla_442359
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
�
H__inference_Output_layer_layer_call_and_return_conditional_losses_473101

inputs4
!tensordot_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
O
#__inference__update_step_xla_442354
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:H D

_output_shapes

:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�
K
#__inference__update_step_xla_442349
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
�

d
E__inference_dropout_7_layer_call_and_return_conditional_losses_471861

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_472115
dense_head_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_head_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_472088o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:���������
*
_user_specified_nameDense_head_input
�
�
(__inference_Dense_3_layer_call_fn_473004

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_Dense_3_layer_call_and_return_conditional_losses_471894t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_dropout_6_layer_call_fn_472911

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_dropout_6_layer_call_and_return_conditional_losses_472002e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_Dense_3_layer_call_and_return_conditional_losses_471894

inputs5
!tensordot_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:z
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0}
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:����������z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
!__inference__wrapped_model_471702
dense_head_inputL
9sequential_1_dense_head_tensordot_readvariableop_resource:	�F
7sequential_1_dense_head_biasadd_readvariableop_resource:	�J
6sequential_1_dense_1_tensordot_readvariableop_resource:
��C
4sequential_1_dense_1_biasadd_readvariableop_resource:	�J
6sequential_1_dense_2_tensordot_readvariableop_resource:
��C
4sequential_1_dense_2_biasadd_readvariableop_resource:	�J
6sequential_1_dense_3_tensordot_readvariableop_resource:
��C
4sequential_1_dense_3_biasadd_readvariableop_resource:	�N
;sequential_1_output_layer_tensordot_readvariableop_resource:	�G
9sequential_1_output_layer_biasadd_readvariableop_resource:J
8sequential_1_final_output_matmul_readvariableop_resource:G
9sequential_1_final_output_biasadd_readvariableop_resource:
identity��+sequential_1/Dense_1/BiasAdd/ReadVariableOp�-sequential_1/Dense_1/Tensordot/ReadVariableOp�+sequential_1/Dense_2/BiasAdd/ReadVariableOp�-sequential_1/Dense_2/Tensordot/ReadVariableOp�+sequential_1/Dense_3/BiasAdd/ReadVariableOp�-sequential_1/Dense_3/Tensordot/ReadVariableOp�.sequential_1/Dense_head/BiasAdd/ReadVariableOp�0sequential_1/Dense_head/Tensordot/ReadVariableOp�0sequential_1/Final_output/BiasAdd/ReadVariableOp�/sequential_1/Final_output/MatMul/ReadVariableOp�0sequential_1/Output_layer/BiasAdd/ReadVariableOp�2sequential_1/Output_layer/Tensordot/ReadVariableOp�
0sequential_1/Dense_head/Tensordot/ReadVariableOpReadVariableOp9sequential_1_dense_head_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0p
&sequential_1/Dense_head/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:w
&sequential_1/Dense_head/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
'sequential_1/Dense_head/Tensordot/ShapeShapedense_head_input*
T0*
_output_shapes
::��q
/sequential_1/Dense_head/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_1/Dense_head/Tensordot/GatherV2GatherV20sequential_1/Dense_head/Tensordot/Shape:output:0/sequential_1/Dense_head/Tensordot/free:output:08sequential_1/Dense_head/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
1sequential_1/Dense_head/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_1/Dense_head/Tensordot/GatherV2_1GatherV20sequential_1/Dense_head/Tensordot/Shape:output:0/sequential_1/Dense_head/Tensordot/axes:output:0:sequential_1/Dense_head/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
'sequential_1/Dense_head/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
&sequential_1/Dense_head/Tensordot/ProdProd3sequential_1/Dense_head/Tensordot/GatherV2:output:00sequential_1/Dense_head/Tensordot/Const:output:0*
T0*
_output_shapes
: s
)sequential_1/Dense_head/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
(sequential_1/Dense_head/Tensordot/Prod_1Prod5sequential_1/Dense_head/Tensordot/GatherV2_1:output:02sequential_1/Dense_head/Tensordot/Const_1:output:0*
T0*
_output_shapes
: o
-sequential_1/Dense_head/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(sequential_1/Dense_head/Tensordot/concatConcatV2/sequential_1/Dense_head/Tensordot/free:output:0/sequential_1/Dense_head/Tensordot/axes:output:06sequential_1/Dense_head/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
'sequential_1/Dense_head/Tensordot/stackPack/sequential_1/Dense_head/Tensordot/Prod:output:01sequential_1/Dense_head/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
+sequential_1/Dense_head/Tensordot/transpose	Transposedense_head_input1sequential_1/Dense_head/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
)sequential_1/Dense_head/Tensordot/ReshapeReshape/sequential_1/Dense_head/Tensordot/transpose:y:00sequential_1/Dense_head/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
(sequential_1/Dense_head/Tensordot/MatMulMatMul2sequential_1/Dense_head/Tensordot/Reshape:output:08sequential_1/Dense_head/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
)sequential_1/Dense_head/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�q
/sequential_1/Dense_head/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_1/Dense_head/Tensordot/concat_1ConcatV23sequential_1/Dense_head/Tensordot/GatherV2:output:02sequential_1/Dense_head/Tensordot/Const_2:output:08sequential_1/Dense_head/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
!sequential_1/Dense_head/TensordotReshape2sequential_1/Dense_head/Tensordot/MatMul:product:03sequential_1/Dense_head/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
.sequential_1/Dense_head/BiasAdd/ReadVariableOpReadVariableOp7sequential_1_dense_head_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/Dense_head/BiasAddBiasAdd*sequential_1/Dense_head/Tensordot:output:06sequential_1/Dense_head/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:�����������
sequential_1/Dense_head/ReluRelu(sequential_1/Dense_head/BiasAdd:output:0*
T0*,
_output_shapes
:�����������
sequential_1/dropout_5/IdentityIdentity*sequential_1/Dense_head/Relu:activations:0*
T0*,
_output_shapes
:�����������
-sequential_1/Dense_1/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0m
#sequential_1/Dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/Dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
$sequential_1/Dense_1/Tensordot/ShapeShape(sequential_1/dropout_5/Identity:output:0*
T0*
_output_shapes
::��n
,sequential_1/Dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/Dense_1/Tensordot/GatherV2GatherV2-sequential_1/Dense_1/Tensordot/Shape:output:0,sequential_1/Dense_1/Tensordot/free:output:05sequential_1/Dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/Dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/Dense_1/Tensordot/GatherV2_1GatherV2-sequential_1/Dense_1/Tensordot/Shape:output:0,sequential_1/Dense_1/Tensordot/axes:output:07sequential_1/Dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/Dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/Dense_1/Tensordot/ProdProd0sequential_1/Dense_1/Tensordot/GatherV2:output:0-sequential_1/Dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/Dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/Dense_1/Tensordot/Prod_1Prod2sequential_1/Dense_1/Tensordot/GatherV2_1:output:0/sequential_1/Dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/Dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/Dense_1/Tensordot/concatConcatV2,sequential_1/Dense_1/Tensordot/free:output:0,sequential_1/Dense_1/Tensordot/axes:output:03sequential_1/Dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/Dense_1/Tensordot/stackPack,sequential_1/Dense_1/Tensordot/Prod:output:0.sequential_1/Dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/Dense_1/Tensordot/transpose	Transpose(sequential_1/dropout_5/Identity:output:0.sequential_1/Dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
&sequential_1/Dense_1/Tensordot/ReshapeReshape,sequential_1/Dense_1/Tensordot/transpose:y:0-sequential_1/Dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/Dense_1/Tensordot/MatMulMatMul/sequential_1/Dense_1/Tensordot/Reshape:output:05sequential_1/Dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
&sequential_1/Dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�n
,sequential_1/Dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/Dense_1/Tensordot/concat_1ConcatV20sequential_1/Dense_1/Tensordot/GatherV2:output:0/sequential_1/Dense_1/Tensordot/Const_2:output:05sequential_1/Dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/Dense_1/TensordotReshape/sequential_1/Dense_1/Tensordot/MatMul:product:00sequential_1/Dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
+sequential_1/Dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/Dense_1/BiasAddBiasAdd'sequential_1/Dense_1/Tensordot:output:03sequential_1/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������
sequential_1/Dense_1/ReluRelu%sequential_1/Dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:�����������
sequential_1/dropout_6/IdentityIdentity'sequential_1/Dense_1/Relu:activations:0*
T0*,
_output_shapes
:�����������
-sequential_1/Dense_2/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0m
#sequential_1/Dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/Dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
$sequential_1/Dense_2/Tensordot/ShapeShape(sequential_1/dropout_6/Identity:output:0*
T0*
_output_shapes
::��n
,sequential_1/Dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/Dense_2/Tensordot/GatherV2GatherV2-sequential_1/Dense_2/Tensordot/Shape:output:0,sequential_1/Dense_2/Tensordot/free:output:05sequential_1/Dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/Dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/Dense_2/Tensordot/GatherV2_1GatherV2-sequential_1/Dense_2/Tensordot/Shape:output:0,sequential_1/Dense_2/Tensordot/axes:output:07sequential_1/Dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/Dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/Dense_2/Tensordot/ProdProd0sequential_1/Dense_2/Tensordot/GatherV2:output:0-sequential_1/Dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/Dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/Dense_2/Tensordot/Prod_1Prod2sequential_1/Dense_2/Tensordot/GatherV2_1:output:0/sequential_1/Dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/Dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/Dense_2/Tensordot/concatConcatV2,sequential_1/Dense_2/Tensordot/free:output:0,sequential_1/Dense_2/Tensordot/axes:output:03sequential_1/Dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/Dense_2/Tensordot/stackPack,sequential_1/Dense_2/Tensordot/Prod:output:0.sequential_1/Dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/Dense_2/Tensordot/transpose	Transpose(sequential_1/dropout_6/Identity:output:0.sequential_1/Dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
&sequential_1/Dense_2/Tensordot/ReshapeReshape,sequential_1/Dense_2/Tensordot/transpose:y:0-sequential_1/Dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/Dense_2/Tensordot/MatMulMatMul/sequential_1/Dense_2/Tensordot/Reshape:output:05sequential_1/Dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
&sequential_1/Dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�n
,sequential_1/Dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/Dense_2/Tensordot/concat_1ConcatV20sequential_1/Dense_2/Tensordot/GatherV2:output:0/sequential_1/Dense_2/Tensordot/Const_2:output:05sequential_1/Dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/Dense_2/TensordotReshape/sequential_1/Dense_2/Tensordot/MatMul:product:00sequential_1/Dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
+sequential_1/Dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/Dense_2/BiasAddBiasAdd'sequential_1/Dense_2/Tensordot:output:03sequential_1/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������
sequential_1/Dense_2/ReluRelu%sequential_1/Dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:�����������
sequential_1/dropout_7/IdentityIdentity'sequential_1/Dense_2/Relu:activations:0*
T0*,
_output_shapes
:�����������
-sequential_1/Dense_3/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0m
#sequential_1/Dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/Dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
$sequential_1/Dense_3/Tensordot/ShapeShape(sequential_1/dropout_7/Identity:output:0*
T0*
_output_shapes
::��n
,sequential_1/Dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/Dense_3/Tensordot/GatherV2GatherV2-sequential_1/Dense_3/Tensordot/Shape:output:0,sequential_1/Dense_3/Tensordot/free:output:05sequential_1/Dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/Dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_1/Dense_3/Tensordot/GatherV2_1GatherV2-sequential_1/Dense_3/Tensordot/Shape:output:0,sequential_1/Dense_3/Tensordot/axes:output:07sequential_1/Dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/Dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_1/Dense_3/Tensordot/ProdProd0sequential_1/Dense_3/Tensordot/GatherV2:output:0-sequential_1/Dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/Dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_1/Dense_3/Tensordot/Prod_1Prod2sequential_1/Dense_3/Tensordot/GatherV2_1:output:0/sequential_1/Dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/Dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_1/Dense_3/Tensordot/concatConcatV2,sequential_1/Dense_3/Tensordot/free:output:0,sequential_1/Dense_3/Tensordot/axes:output:03sequential_1/Dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_1/Dense_3/Tensordot/stackPack,sequential_1/Dense_3/Tensordot/Prod:output:0.sequential_1/Dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_1/Dense_3/Tensordot/transpose	Transpose(sequential_1/dropout_7/Identity:output:0.sequential_1/Dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
&sequential_1/Dense_3/Tensordot/ReshapeReshape,sequential_1/Dense_3/Tensordot/transpose:y:0-sequential_1/Dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_1/Dense_3/Tensordot/MatMulMatMul/sequential_1/Dense_3/Tensordot/Reshape:output:05sequential_1/Dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
&sequential_1/Dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�n
,sequential_1/Dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_1/Dense_3/Tensordot/concat_1ConcatV20sequential_1/Dense_3/Tensordot/GatherV2:output:0/sequential_1/Dense_3/Tensordot/Const_2:output:05sequential_1/Dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_1/Dense_3/TensordotReshape/sequential_1/Dense_3/Tensordot/MatMul:product:00sequential_1/Dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
+sequential_1/Dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1/Dense_3/BiasAddBiasAdd'sequential_1/Dense_3/Tensordot:output:03sequential_1/Dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������
sequential_1/Dense_3/ReluRelu%sequential_1/Dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:�����������
sequential_1/dropout_8/IdentityIdentity'sequential_1/Dense_3/Relu:activations:0*
T0*,
_output_shapes
:�����������
2sequential_1/Output_layer/Tensordot/ReadVariableOpReadVariableOp;sequential_1_output_layer_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0r
(sequential_1/Output_layer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:y
(sequential_1/Output_layer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
)sequential_1/Output_layer/Tensordot/ShapeShape(sequential_1/dropout_8/Identity:output:0*
T0*
_output_shapes
::��s
1sequential_1/Output_layer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_1/Output_layer/Tensordot/GatherV2GatherV22sequential_1/Output_layer/Tensordot/Shape:output:01sequential_1/Output_layer/Tensordot/free:output:0:sequential_1/Output_layer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:u
3sequential_1/Output_layer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
.sequential_1/Output_layer/Tensordot/GatherV2_1GatherV22sequential_1/Output_layer/Tensordot/Shape:output:01sequential_1/Output_layer/Tensordot/axes:output:0<sequential_1/Output_layer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:s
)sequential_1/Output_layer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
(sequential_1/Output_layer/Tensordot/ProdProd5sequential_1/Output_layer/Tensordot/GatherV2:output:02sequential_1/Output_layer/Tensordot/Const:output:0*
T0*
_output_shapes
: u
+sequential_1/Output_layer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
*sequential_1/Output_layer/Tensordot/Prod_1Prod7sequential_1/Output_layer/Tensordot/GatherV2_1:output:04sequential_1/Output_layer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: q
/sequential_1/Output_layer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*sequential_1/Output_layer/Tensordot/concatConcatV21sequential_1/Output_layer/Tensordot/free:output:01sequential_1/Output_layer/Tensordot/axes:output:08sequential_1/Output_layer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)sequential_1/Output_layer/Tensordot/stackPack1sequential_1/Output_layer/Tensordot/Prod:output:03sequential_1/Output_layer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
-sequential_1/Output_layer/Tensordot/transpose	Transpose(sequential_1/dropout_8/Identity:output:03sequential_1/Output_layer/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
+sequential_1/Output_layer/Tensordot/ReshapeReshape1sequential_1/Output_layer/Tensordot/transpose:y:02sequential_1/Output_layer/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
*sequential_1/Output_layer/Tensordot/MatMulMatMul4sequential_1/Output_layer/Tensordot/Reshape:output:0:sequential_1/Output_layer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������u
+sequential_1/Output_layer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:s
1sequential_1/Output_layer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
,sequential_1/Output_layer/Tensordot/concat_1ConcatV25sequential_1/Output_layer/Tensordot/GatherV2:output:04sequential_1/Output_layer/Tensordot/Const_2:output:0:sequential_1/Output_layer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
#sequential_1/Output_layer/TensordotReshape4sequential_1/Output_layer/Tensordot/MatMul:product:05sequential_1/Output_layer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
0sequential_1/Output_layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_1/Output_layer/BiasAddBiasAdd,sequential_1/Output_layer/Tensordot:output:08sequential_1/Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������k
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
sequential_1/flatten/ReshapeReshape*sequential_1/Output_layer/BiasAdd:output:0#sequential_1/flatten/Const:output:0*
T0*'
_output_shapes
:����������
/sequential_1/Final_output/MatMul/ReadVariableOpReadVariableOp8sequential_1_final_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
 sequential_1/Final_output/MatMulMatMul%sequential_1/flatten/Reshape:output:07sequential_1/Final_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0sequential_1/Final_output/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_final_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!sequential_1/Final_output/BiasAddBiasAdd*sequential_1/Final_output/MatMul:product:08sequential_1/Final_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*sequential_1/Final_output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^sequential_1/Dense_1/BiasAdd/ReadVariableOp.^sequential_1/Dense_1/Tensordot/ReadVariableOp,^sequential_1/Dense_2/BiasAdd/ReadVariableOp.^sequential_1/Dense_2/Tensordot/ReadVariableOp,^sequential_1/Dense_3/BiasAdd/ReadVariableOp.^sequential_1/Dense_3/Tensordot/ReadVariableOp/^sequential_1/Dense_head/BiasAdd/ReadVariableOp1^sequential_1/Dense_head/Tensordot/ReadVariableOp1^sequential_1/Final_output/BiasAdd/ReadVariableOp0^sequential_1/Final_output/MatMul/ReadVariableOp1^sequential_1/Output_layer/BiasAdd/ReadVariableOp3^sequential_1/Output_layer/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2Z
+sequential_1/Dense_1/BiasAdd/ReadVariableOp+sequential_1/Dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/Dense_1/Tensordot/ReadVariableOp-sequential_1/Dense_1/Tensordot/ReadVariableOp2Z
+sequential_1/Dense_2/BiasAdd/ReadVariableOp+sequential_1/Dense_2/BiasAdd/ReadVariableOp2^
-sequential_1/Dense_2/Tensordot/ReadVariableOp-sequential_1/Dense_2/Tensordot/ReadVariableOp2Z
+sequential_1/Dense_3/BiasAdd/ReadVariableOp+sequential_1/Dense_3/BiasAdd/ReadVariableOp2^
-sequential_1/Dense_3/Tensordot/ReadVariableOp-sequential_1/Dense_3/Tensordot/ReadVariableOp2`
.sequential_1/Dense_head/BiasAdd/ReadVariableOp.sequential_1/Dense_head/BiasAdd/ReadVariableOp2d
0sequential_1/Dense_head/Tensordot/ReadVariableOp0sequential_1/Dense_head/Tensordot/ReadVariableOp2d
0sequential_1/Final_output/BiasAdd/ReadVariableOp0sequential_1/Final_output/BiasAdd/ReadVariableOp2b
/sequential_1/Final_output/MatMul/ReadVariableOp/sequential_1/Final_output/MatMul/ReadVariableOp2d
0sequential_1/Output_layer/BiasAdd/ReadVariableOp0sequential_1/Output_layer/BiasAdd/ReadVariableOp2h
2sequential_1/Output_layer/Tensordot/ReadVariableOp2sequential_1/Output_layer/Tensordot/ReadVariableOp:] Y
+
_output_shapes
:���������
*
_user_specified_nameDense_head_input
�	
�
H__inference_Final_output_layer_call_and_return_conditional_losses_473131

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�

H__inference_sequential_1_layer_call_and_return_conditional_losses_472790

inputs?
,dense_head_tensordot_readvariableop_resource:	�9
*dense_head_biasadd_readvariableop_resource:	�=
)dense_1_tensordot_readvariableop_resource:
��6
'dense_1_biasadd_readvariableop_resource:	�=
)dense_2_tensordot_readvariableop_resource:
��6
'dense_2_biasadd_readvariableop_resource:	�=
)dense_3_tensordot_readvariableop_resource:
��6
'dense_3_biasadd_readvariableop_resource:	�A
.output_layer_tensordot_readvariableop_resource:	�:
,output_layer_biasadd_readvariableop_resource:=
+final_output_matmul_readvariableop_resource::
,final_output_biasadd_readvariableop_resource:
identity��Dense_1/BiasAdd/ReadVariableOp� Dense_1/Tensordot/ReadVariableOp�Dense_2/BiasAdd/ReadVariableOp� Dense_2/Tensordot/ReadVariableOp�Dense_3/BiasAdd/ReadVariableOp� Dense_3/Tensordot/ReadVariableOp�!Dense_head/BiasAdd/ReadVariableOp�#Dense_head/Tensordot/ReadVariableOp�3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp�#Final_output/BiasAdd/ReadVariableOp�"Final_output/MatMul/ReadVariableOp�#Output_layer/BiasAdd/ReadVariableOp�%Output_layer/Tensordot/ReadVariableOp�
#Dense_head/Tensordot/ReadVariableOpReadVariableOp,dense_head_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0c
Dense_head/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:j
Dense_head/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ^
Dense_head/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��d
"Dense_head/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/GatherV2GatherV2#Dense_head/Tensordot/Shape:output:0"Dense_head/Tensordot/free:output:0+Dense_head/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:f
$Dense_head/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/GatherV2_1GatherV2#Dense_head/Tensordot/Shape:output:0"Dense_head/Tensordot/axes:output:0-Dense_head/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
Dense_head/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_head/Tensordot/ProdProd&Dense_head/Tensordot/GatherV2:output:0#Dense_head/Tensordot/Const:output:0*
T0*
_output_shapes
: f
Dense_head/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_head/Tensordot/Prod_1Prod(Dense_head/Tensordot/GatherV2_1:output:0%Dense_head/Tensordot/Const_1:output:0*
T0*
_output_shapes
: b
 Dense_head/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/concatConcatV2"Dense_head/Tensordot/free:output:0"Dense_head/Tensordot/axes:output:0)Dense_head/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_head/Tensordot/stackPack"Dense_head/Tensordot/Prod:output:0$Dense_head/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_head/Tensordot/transpose	Transposeinputs$Dense_head/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
Dense_head/Tensordot/ReshapeReshape"Dense_head/Tensordot/transpose:y:0#Dense_head/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_head/Tensordot/MatMulMatMul%Dense_head/Tensordot/Reshape:output:0+Dense_head/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������g
Dense_head/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�d
"Dense_head/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_head/Tensordot/concat_1ConcatV2&Dense_head/Tensordot/GatherV2:output:0%Dense_head/Tensordot/Const_2:output:0+Dense_head/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_head/TensordotReshape%Dense_head/Tensordot/MatMul:product:0&Dense_head/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
!Dense_head/BiasAdd/ReadVariableOpReadVariableOp*dense_head_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_head/BiasAddBiasAddDense_head/Tensordot:output:0)Dense_head/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������k
Dense_head/ReluReluDense_head/BiasAdd:output:0*
T0*,
_output_shapes
:����������t
dropout_5/IdentityIdentityDense_head/Relu:activations:0*
T0*,
_output_shapes
:�����������
 Dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
Dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
Dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
Dense_1/Tensordot/ShapeShapedropout_5/Identity:output:0*
T0*
_output_shapes
::��a
Dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/GatherV2GatherV2 Dense_1/Tensordot/Shape:output:0Dense_1/Tensordot/free:output:0(Dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!Dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/GatherV2_1GatherV2 Dense_1/Tensordot/Shape:output:0Dense_1/Tensordot/axes:output:0*Dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
Dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_1/Tensordot/ProdProd#Dense_1/Tensordot/GatherV2:output:0 Dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: c
Dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_1/Tensordot/Prod_1Prod%Dense_1/Tensordot/GatherV2_1:output:0"Dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
Dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/concatConcatV2Dense_1/Tensordot/free:output:0Dense_1/Tensordot/axes:output:0&Dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_1/Tensordot/stackPackDense_1/Tensordot/Prod:output:0!Dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_1/Tensordot/transpose	Transposedropout_5/Identity:output:0!Dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Dense_1/Tensordot/ReshapeReshapeDense_1/Tensordot/transpose:y:0 Dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_1/Tensordot/MatMulMatMul"Dense_1/Tensordot/Reshape:output:0(Dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
Dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
Dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_1/Tensordot/concat_1ConcatV2#Dense_1/Tensordot/GatherV2:output:0"Dense_1/Tensordot/Const_2:output:0(Dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_1/TensordotReshape"Dense_1/Tensordot/MatMul:product:0#Dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_1/BiasAddBiasAddDense_1/Tensordot:output:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������q
dropout_6/IdentityIdentityDense_1/Relu:activations:0*
T0*,
_output_shapes
:�����������
 Dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
Dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
Dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
Dense_2/Tensordot/ShapeShapedropout_6/Identity:output:0*
T0*
_output_shapes
::��a
Dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/GatherV2GatherV2 Dense_2/Tensordot/Shape:output:0Dense_2/Tensordot/free:output:0(Dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!Dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/GatherV2_1GatherV2 Dense_2/Tensordot/Shape:output:0Dense_2/Tensordot/axes:output:0*Dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
Dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_2/Tensordot/ProdProd#Dense_2/Tensordot/GatherV2:output:0 Dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: c
Dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_2/Tensordot/Prod_1Prod%Dense_2/Tensordot/GatherV2_1:output:0"Dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
Dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/concatConcatV2Dense_2/Tensordot/free:output:0Dense_2/Tensordot/axes:output:0&Dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_2/Tensordot/stackPackDense_2/Tensordot/Prod:output:0!Dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_2/Tensordot/transpose	Transposedropout_6/Identity:output:0!Dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Dense_2/Tensordot/ReshapeReshapeDense_2/Tensordot/transpose:y:0 Dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_2/Tensordot/MatMulMatMul"Dense_2/Tensordot/Reshape:output:0(Dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
Dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
Dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_2/Tensordot/concat_1ConcatV2#Dense_2/Tensordot/GatherV2:output:0"Dense_2/Tensordot/Const_2:output:0(Dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_2/TensordotReshape"Dense_2/Tensordot/MatMul:product:0#Dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_2/BiasAddBiasAddDense_2/Tensordot:output:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
Dense_2/ReluReluDense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������q
dropout_7/IdentityIdentityDense_2/Relu:activations:0*
T0*,
_output_shapes
:�����������
 Dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
��*
dtype0`
Dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
Dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       p
Dense_3/Tensordot/ShapeShapedropout_7/Identity:output:0*
T0*
_output_shapes
::��a
Dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/GatherV2GatherV2 Dense_3/Tensordot/Shape:output:0Dense_3/Tensordot/free:output:0(Dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!Dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/GatherV2_1GatherV2 Dense_3/Tensordot/Shape:output:0Dense_3/Tensordot/axes:output:0*Dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
Dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Dense_3/Tensordot/ProdProd#Dense_3/Tensordot/GatherV2:output:0 Dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: c
Dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Dense_3/Tensordot/Prod_1Prod%Dense_3/Tensordot/GatherV2_1:output:0"Dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
Dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/concatConcatV2Dense_3/Tensordot/free:output:0Dense_3/Tensordot/axes:output:0&Dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_3/Tensordot/stackPackDense_3/Tensordot/Prod:output:0!Dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Dense_3/Tensordot/transpose	Transposedropout_7/Identity:output:0!Dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Dense_3/Tensordot/ReshapeReshapeDense_3/Tensordot/transpose:y:0 Dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Dense_3/Tensordot/MatMulMatMul"Dense_3/Tensordot/Reshape:output:0(Dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������d
Dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�a
Dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Dense_3/Tensordot/concat_1ConcatV2#Dense_3/Tensordot/GatherV2:output:0"Dense_3/Tensordot/Const_2:output:0(Dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Dense_3/TensordotReshape"Dense_3/Tensordot/MatMul:product:0#Dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:�����������
Dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Dense_3/BiasAddBiasAddDense_3/Tensordot:output:0&Dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������e
Dense_3/ReluReluDense_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������q
dropout_8/IdentityIdentityDense_3/Relu:activations:0*
T0*,
_output_shapes
:�����������
%Output_layer/Tensordot/ReadVariableOpReadVariableOp.output_layer_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0e
Output_layer/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
Output_layer/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       u
Output_layer/Tensordot/ShapeShapedropout_8/Identity:output:0*
T0*
_output_shapes
::��f
$Output_layer/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_layer/Tensordot/GatherV2GatherV2%Output_layer/Tensordot/Shape:output:0$Output_layer/Tensordot/free:output:0-Output_layer/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:h
&Output_layer/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
!Output_layer/Tensordot/GatherV2_1GatherV2%Output_layer/Tensordot/Shape:output:0$Output_layer/Tensordot/axes:output:0/Output_layer/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:f
Output_layer/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
Output_layer/Tensordot/ProdProd(Output_layer/Tensordot/GatherV2:output:0%Output_layer/Tensordot/Const:output:0*
T0*
_output_shapes
: h
Output_layer/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
Output_layer/Tensordot/Prod_1Prod*Output_layer/Tensordot/GatherV2_1:output:0'Output_layer/Tensordot/Const_1:output:0*
T0*
_output_shapes
: d
"Output_layer/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_layer/Tensordot/concatConcatV2$Output_layer/Tensordot/free:output:0$Output_layer/Tensordot/axes:output:0+Output_layer/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
Output_layer/Tensordot/stackPack$Output_layer/Tensordot/Prod:output:0&Output_layer/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
 Output_layer/Tensordot/transpose	Transposedropout_8/Identity:output:0&Output_layer/Tensordot/concat:output:0*
T0*,
_output_shapes
:�����������
Output_layer/Tensordot/ReshapeReshape$Output_layer/Tensordot/transpose:y:0%Output_layer/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Output_layer/Tensordot/MatMulMatMul'Output_layer/Tensordot/Reshape:output:0-Output_layer/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
Output_layer/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:f
$Output_layer/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Output_layer/Tensordot/concat_1ConcatV2(Output_layer/Tensordot/GatherV2:output:0'Output_layer/Tensordot/Const_2:output:0-Output_layer/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Output_layer/TensordotReshape'Output_layer/Tensordot/MatMul:product:0(Output_layer/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:����������
#Output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Output_layer/BiasAddBiasAddOutput_layer/Tensordot:output:0+Output_layer/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
flatten/ReshapeReshapeOutput_layer/BiasAdd:output:0flatten/Const:output:0*
T0*'
_output_shapes
:����������
"Final_output/MatMul/ReadVariableOpReadVariableOp+final_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
Final_output/MatMulMatMulflatten/Reshape:output:0*Final_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#Final_output/BiasAdd/ReadVariableOpReadVariableOp,final_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
Final_output/BiasAddBiasAddFinal_output/MatMul:product:0+Final_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp,dense_head_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0�
$Dense_head/kernel/Regularizer/L2LossL2Loss;Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#Dense_head/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
!Dense_head/kernel/Regularizer/mulMul,Dense_head/kernel/Regularizer/mul/x:output:0-Dense_head/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: l
IdentityIdentityFinal_output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^Dense_1/BiasAdd/ReadVariableOp!^Dense_1/Tensordot/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp!^Dense_2/Tensordot/ReadVariableOp^Dense_3/BiasAdd/ReadVariableOp!^Dense_3/Tensordot/ReadVariableOp"^Dense_head/BiasAdd/ReadVariableOp$^Dense_head/Tensordot/ReadVariableOp4^Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp$^Final_output/BiasAdd/ReadVariableOp#^Final_output/MatMul/ReadVariableOp$^Output_layer/BiasAdd/ReadVariableOp&^Output_layer/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2D
 Dense_1/Tensordot/ReadVariableOp Dense_1/Tensordot/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2D
 Dense_2/Tensordot/ReadVariableOp Dense_2/Tensordot/ReadVariableOp2@
Dense_3/BiasAdd/ReadVariableOpDense_3/BiasAdd/ReadVariableOp2D
 Dense_3/Tensordot/ReadVariableOp Dense_3/Tensordot/ReadVariableOp2F
!Dense_head/BiasAdd/ReadVariableOp!Dense_head/BiasAdd/ReadVariableOp2J
#Dense_head/Tensordot/ReadVariableOp#Dense_head/Tensordot/ReadVariableOp2j
3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp3Dense_head/kernel/Regularizer/L2Loss/ReadVariableOp2J
#Final_output/BiasAdd/ReadVariableOp#Final_output/BiasAdd/ReadVariableOp2H
"Final_output/MatMul/ReadVariableOp"Final_output/MatMul/ReadVariableOp2J
#Output_layer/BiasAdd/ReadVariableOp#Output_layer/BiasAdd/ReadVariableOp2N
%Output_layer/Tensordot/ReadVariableOp%Output_layer/Tensordot/ReadVariableOp:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

d
E__inference_dropout_6_layer_call_and_return_conditional_losses_471810

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_472024

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:����������`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_473057

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*,
_output_shapes
:����������f
IdentityIdentitydropout/SelectV2:output:0*
T0*,
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_Dense_head_layer_call_fn_472799

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_Dense_head_layer_call_and_return_conditional_losses_471741t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
-__inference_sequential_1_layer_call_fn_472187
dense_head_input
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:
��
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:
	unknown_9:

unknown_10:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_head_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_472160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
+
_output_shapes
:���������
*
_user_specified_nameDense_head_input"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Q
Dense_head_input=
"serving_default_Dense_head_input:0���������@
Final_output0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
#_random_generator"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses
2_random_generator"
_tf_keras_layer
�
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

9kernel
:bias"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses

Hkernel
Ibias"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
P_random_generator"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

Wkernel
Xbias"
_tf_keras_layer
�
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
�
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses

ekernel
fbias"
_tf_keras_layer
v
0
1
*2
+3
94
:5
H6
I7
W8
X9
e10
f11"
trackable_list_wrapper
v
0
1
*2
+3
94
:5
H6
I7
W8
X9
e10
f11"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
�
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
mtrace_0
ntrace_1
otrace_2
ptrace_32�
-__inference_sequential_1_layer_call_fn_472115
-__inference_sequential_1_layer_call_fn_472187
-__inference_sequential_1_layer_call_fn_472425
-__inference_sequential_1_layer_call_fn_472454�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zmtrace_0zntrace_1zotrace_2zptrace_3
�
qtrace_0
rtrace_1
strace_2
ttrace_32�
H__inference_sequential_1_layer_call_and_return_conditional_losses_471979
H__inference_sequential_1_layer_call_and_return_conditional_losses_472042
H__inference_sequential_1_layer_call_and_return_conditional_losses_472636
H__inference_sequential_1_layer_call_and_return_conditional_losses_472790�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zqtrace_0zrtrace_1zstrace_2zttrace_3
�B�
!__inference__wrapped_model_471702Dense_head_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
u
_variables
v_iterations
w_learning_rate
x_index_dict
y
_momentums
z_velocities
{_update_step_xla"
experimentalOptimizer
,
|serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_Dense_head_layer_call_fn_472799�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_Dense_head_layer_call_and_return_conditional_losses_472834�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"	�2Dense_head/kernel
:�2Dense_head/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_5_layer_call_fn_472839
*__inference_dropout_5_layer_call_fn_472844�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_5_layer_call_and_return_conditional_losses_472856
E__inference_dropout_5_layer_call_and_return_conditional_losses_472861�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_Dense_1_layer_call_fn_472870�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_Dense_1_layer_call_and_return_conditional_losses_472901�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2Dense_1/kernel
:�2Dense_1/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_6_layer_call_fn_472906
*__inference_dropout_6_layer_call_fn_472911�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_6_layer_call_and_return_conditional_losses_472923
E__inference_dropout_6_layer_call_and_return_conditional_losses_472928�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_Dense_2_layer_call_fn_472937�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_Dense_2_layer_call_and_return_conditional_losses_472968�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2Dense_2/kernel
:�2Dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_7_layer_call_fn_472973
*__inference_dropout_7_layer_call_fn_472978�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_7_layer_call_and_return_conditional_losses_472990
E__inference_dropout_7_layer_call_and_return_conditional_losses_472995�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_Dense_3_layer_call_fn_473004�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_Dense_3_layer_call_and_return_conditional_losses_473035�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 
��2Dense_3/kernel
:�2Dense_3/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
*__inference_dropout_8_layer_call_fn_473040
*__inference_dropout_8_layer_call_fn_473045�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
E__inference_dropout_8_layer_call_and_return_conditional_losses_473057
E__inference_dropout_8_layer_call_and_return_conditional_losses_473062�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_Output_layer_layer_call_fn_473071�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_Output_layer_layer_call_and_return_conditional_losses_473101�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
&:$	�2Output_layer/kernel
:2Output_layer/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Y	variables
Ztrainable_variables
[regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_473106�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_473112�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_Final_output_layer_call_fn_473121�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_Final_output_layer_call_and_return_conditional_losses_473131�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
%:#2Final_output/kernel
:2Final_output/bias
�
�trace_02�
__inference_loss_fn_0_473140�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� z�trace_0
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_sequential_1_layer_call_fn_472115Dense_head_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_472187Dense_head_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_472425inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_sequential_1_layer_call_fn_472454inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_471979Dense_head_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_472042Dense_head_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_472636inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_sequential_1_layer_call_and_return_conditional_losses_472790inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
v0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11"
trackable_list_wrapper
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_112�
#__inference__update_step_xla_442304
#__inference__update_step_xla_442309
#__inference__update_step_xla_442314
#__inference__update_step_xla_442319
#__inference__update_step_xla_442324
#__inference__update_step_xla_442329
#__inference__update_step_xla_442334
#__inference__update_step_xla_442339
#__inference__update_step_xla_442344
#__inference__update_step_xla_442349
#__inference__update_step_xla_442354
#__inference__update_step_xla_442359�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11
�B�
$__inference_signature_wrapper_472392Dense_head_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_Dense_head_layer_call_fn_472799inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_Dense_head_layer_call_and_return_conditional_losses_472834inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_5_layer_call_fn_472839inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_5_layer_call_fn_472844inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_5_layer_call_and_return_conditional_losses_472856inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_5_layer_call_and_return_conditional_losses_472861inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_Dense_1_layer_call_fn_472870inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_Dense_1_layer_call_and_return_conditional_losses_472901inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_6_layer_call_fn_472906inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_6_layer_call_fn_472911inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_6_layer_call_and_return_conditional_losses_472923inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_6_layer_call_and_return_conditional_losses_472928inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_Dense_2_layer_call_fn_472937inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_Dense_2_layer_call_and_return_conditional_losses_472968inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_7_layer_call_fn_472973inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_7_layer_call_fn_472978inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_7_layer_call_and_return_conditional_losses_472990inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_7_layer_call_and_return_conditional_losses_472995inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_Dense_3_layer_call_fn_473004inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_Dense_3_layer_call_and_return_conditional_losses_473035inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_dropout_8_layer_call_fn_473040inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_dropout_8_layer_call_fn_473045inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_8_layer_call_and_return_conditional_losses_473057inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_dropout_8_layer_call_and_return_conditional_losses_473062inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_Output_layer_layer_call_fn_473071inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Output_layer_layer_call_and_return_conditional_losses_473101inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_flatten_layer_call_fn_473106inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_473112inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
-__inference_Final_output_layer_call_fn_473121inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_Final_output_layer_call_and_return_conditional_losses_473131inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_473140"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
):'	�2Adam/m/Dense_head/kernel
):'	�2Adam/v/Dense_head/kernel
#:!�2Adam/m/Dense_head/bias
#:!�2Adam/v/Dense_head/bias
':%
��2Adam/m/Dense_1/kernel
':%
��2Adam/v/Dense_1/kernel
 :�2Adam/m/Dense_1/bias
 :�2Adam/v/Dense_1/bias
':%
��2Adam/m/Dense_2/kernel
':%
��2Adam/v/Dense_2/kernel
 :�2Adam/m/Dense_2/bias
 :�2Adam/v/Dense_2/bias
':%
��2Adam/m/Dense_3/kernel
':%
��2Adam/v/Dense_3/kernel
 :�2Adam/m/Dense_3/bias
 :�2Adam/v/Dense_3/bias
+:)	�2Adam/m/Output_layer/kernel
+:)	�2Adam/v/Output_layer/kernel
$:"2Adam/m/Output_layer/bias
$:"2Adam/v/Output_layer/bias
*:(2Adam/m/Final_output/kernel
*:(2Adam/v/Final_output/kernel
$:"2Adam/m/Final_output/bias
$:"2Adam/v/Final_output/bias
�B�
#__inference__update_step_xla_442304gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442309gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442314gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442319gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442324gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442329gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442334gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442339gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442344gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442349gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442354gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_442359gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
C__inference_Dense_1_layer_call_and_return_conditional_losses_472901m*+4�1
*�'
%�"
inputs����������
� "1�.
'�$
tensor_0����������
� �
(__inference_Dense_1_layer_call_fn_472870b*+4�1
*�'
%�"
inputs����������
� "&�#
unknown�����������
C__inference_Dense_2_layer_call_and_return_conditional_losses_472968m9:4�1
*�'
%�"
inputs����������
� "1�.
'�$
tensor_0����������
� �
(__inference_Dense_2_layer_call_fn_472937b9:4�1
*�'
%�"
inputs����������
� "&�#
unknown�����������
C__inference_Dense_3_layer_call_and_return_conditional_losses_473035mHI4�1
*�'
%�"
inputs����������
� "1�.
'�$
tensor_0����������
� �
(__inference_Dense_3_layer_call_fn_473004bHI4�1
*�'
%�"
inputs����������
� "&�#
unknown�����������
F__inference_Dense_head_layer_call_and_return_conditional_losses_472834l3�0
)�&
$�!
inputs���������
� "1�.
'�$
tensor_0����������
� �
+__inference_Dense_head_layer_call_fn_472799a3�0
)�&
$�!
inputs���������
� "&�#
unknown�����������
H__inference_Final_output_layer_call_and_return_conditional_losses_473131cef/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
-__inference_Final_output_layer_call_fn_473121Xef/�,
%�"
 �
inputs���������
� "!�
unknown����������
H__inference_Output_layer_layer_call_and_return_conditional_losses_473101lWX4�1
*�'
%�"
inputs����������
� "0�-
&�#
tensor_0���������
� �
-__inference_Output_layer_layer_call_fn_473071aWX4�1
*�'
%�"
inputs����������
� "%�"
unknown����������
#__inference__update_step_xla_442304pj�g
`�]
�
gradient	�
5�2	�
�	�
�
p
` VariableSpec 
`�ժ���?
� "
 �
#__inference__update_step_xla_442309hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`���ڟ�?
� "
 �
#__inference__update_step_xla_442314zt�q
j�g
#� 
gradient����������
6�3	�
�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_442319hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_442324zt�q
j�g
#� 
gradient����������
6�3	�
�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_442329hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_442334zt�q
j�g
#� 
gradient����������
6�3	�
�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_442339hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`�⪀��?
� "
 �
#__inference__update_step_xla_442344xr�o
h�e
"�
gradient���������
5�2	�
�	�
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_442349f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`༖���?
� "
 �
#__inference__update_step_xla_442354nh�e
^�[
�
gradient
4�1	�
�
�
p
` VariableSpec 
`��ݏ��?
� "
 �
#__inference__update_step_xla_442359f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
!__inference__wrapped_model_471702�*+9:HIWXef=�:
3�0
.�+
Dense_head_input���������
� ";�8
6
Final_output&�#
final_output����������
E__inference_dropout_5_layer_call_and_return_conditional_losses_472856m8�5
.�+
%�"
inputs����������
p
� "1�.
'�$
tensor_0����������
� �
E__inference_dropout_5_layer_call_and_return_conditional_losses_472861m8�5
.�+
%�"
inputs����������
p 
� "1�.
'�$
tensor_0����������
� �
*__inference_dropout_5_layer_call_fn_472839b8�5
.�+
%�"
inputs����������
p
� "&�#
unknown�����������
*__inference_dropout_5_layer_call_fn_472844b8�5
.�+
%�"
inputs����������
p 
� "&�#
unknown�����������
E__inference_dropout_6_layer_call_and_return_conditional_losses_472923m8�5
.�+
%�"
inputs����������
p
� "1�.
'�$
tensor_0����������
� �
E__inference_dropout_6_layer_call_and_return_conditional_losses_472928m8�5
.�+
%�"
inputs����������
p 
� "1�.
'�$
tensor_0����������
� �
*__inference_dropout_6_layer_call_fn_472906b8�5
.�+
%�"
inputs����������
p
� "&�#
unknown�����������
*__inference_dropout_6_layer_call_fn_472911b8�5
.�+
%�"
inputs����������
p 
� "&�#
unknown�����������
E__inference_dropout_7_layer_call_and_return_conditional_losses_472990m8�5
.�+
%�"
inputs����������
p
� "1�.
'�$
tensor_0����������
� �
E__inference_dropout_7_layer_call_and_return_conditional_losses_472995m8�5
.�+
%�"
inputs����������
p 
� "1�.
'�$
tensor_0����������
� �
*__inference_dropout_7_layer_call_fn_472973b8�5
.�+
%�"
inputs����������
p
� "&�#
unknown�����������
*__inference_dropout_7_layer_call_fn_472978b8�5
.�+
%�"
inputs����������
p 
� "&�#
unknown�����������
E__inference_dropout_8_layer_call_and_return_conditional_losses_473057m8�5
.�+
%�"
inputs����������
p
� "1�.
'�$
tensor_0����������
� �
E__inference_dropout_8_layer_call_and_return_conditional_losses_473062m8�5
.�+
%�"
inputs����������
p 
� "1�.
'�$
tensor_0����������
� �
*__inference_dropout_8_layer_call_fn_473040b8�5
.�+
%�"
inputs����������
p
� "&�#
unknown�����������
*__inference_dropout_8_layer_call_fn_473045b8�5
.�+
%�"
inputs����������
p 
� "&�#
unknown�����������
C__inference_flatten_layer_call_and_return_conditional_losses_473112c3�0
)�&
$�!
inputs���������
� ",�)
"�
tensor_0���������
� �
(__inference_flatten_layer_call_fn_473106X3�0
)�&
$�!
inputs���������
� "!�
unknown���������D
__inference_loss_fn_0_473140$�

� 
� "�
unknown �
H__inference_sequential_1_layer_call_and_return_conditional_losses_471979�*+9:HIWXefE�B
;�8
.�+
Dense_head_input���������
p

 
� ",�)
"�
tensor_0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_472042�*+9:HIWXefE�B
;�8
.�+
Dense_head_input���������
p 

 
� ",�)
"�
tensor_0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_472636y*+9:HIWXef;�8
1�.
$�!
inputs���������
p

 
� ",�)
"�
tensor_0���������
� �
H__inference_sequential_1_layer_call_and_return_conditional_losses_472790y*+9:HIWXef;�8
1�.
$�!
inputs���������
p 

 
� ",�)
"�
tensor_0���������
� �
-__inference_sequential_1_layer_call_fn_472115x*+9:HIWXefE�B
;�8
.�+
Dense_head_input���������
p

 
� "!�
unknown����������
-__inference_sequential_1_layer_call_fn_472187x*+9:HIWXefE�B
;�8
.�+
Dense_head_input���������
p 

 
� "!�
unknown����������
-__inference_sequential_1_layer_call_fn_472425n*+9:HIWXef;�8
1�.
$�!
inputs���������
p

 
� "!�
unknown����������
-__inference_sequential_1_layer_call_fn_472454n*+9:HIWXef;�8
1�.
$�!
inputs���������
p 

 
� "!�
unknown����������
$__inference_signature_wrapper_472392�*+9:HIWXefQ�N
� 
G�D
B
Dense_head_input.�+
dense_head_input���������";�8
6
Final_output&�#
final_output���������