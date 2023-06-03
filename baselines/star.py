import tensorflow as tf


def deepfm_din(features, mode, params):
    
    # ============= input =============
    # (feature_indices = feature_values = [batch_size, field_size], labels = [batch_size, 1])

    feature, domain_indicator = combine_feature(features, mode, params)

    net = feature 
    # ============= star part =============
    net = star(net, list([256,32]), domain_indicator, params, 'deepfm_din')

    net = tf.compat.v1.layers.dense(inputs=net, units=1,
				   kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
				   kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
				   name='deepfm_din_dense_layer_final')
    output = tf.sigmoid(net, name='out')

    return output

def combine_feature(features, mode, params):
    feature_indices, feature_values = features['feature_indices'], features['feature_values']

    domain_indicator = tf.gather(params=feature_indices, indices=[params['domain_indicator']], axis=1)

    itemid_indices = tf.gather(params=feature_indices, indices=params['index.albumid'], axis=1)
    itemid_values = tf.gather(params=feature_values, indices=params['index.albumid'], axis=1)

    channelid_indices = tf.gather(params=feature_indices, indices=params['index.channelid'], axis=1)
    channelid_values = tf.gather(params=feature_values, indices=params['index.channelid'], axis=1)

    deepfm_indices = tf.gather(params=feature_indices, indices=params['index.deepfm'], axis=1)
    deepfm_values = tf.gather(params=feature_values, indices=params['index.deepfm'], axis=1)

    click_behavior_itemid_indices = tf.gather(params=feature_indices, indices=params['index.play_album_7d_seq'], axis=1)
    click_behavior_itemid_values = tf.gather(params=feature_values, indices=params['index.play_album_7d_seq'], axis=1)
     
    click_behavior_channelid_indices = tf.gather(params=feature_indices, indices=params['index.play_channel_7d_seq'], axis=1)
    click_behavior_channelid_values = tf.gather(params=feature_values, indices=params['index.play_channel_7d_seq'], axis=1)

    click_behavior_clicknum_indices = tf.gather(params=feature_indices, indices=params['index.play_cnt_7d_seq'], axis=1)
    click_behavior_clicknum_values = tf.gather(params=feature_values, indices=params['index.play_cnt_7d_seq'], axis=1)

    w2v_embedding_matrix = tf.Variable(params["w2v_vec"], trainable=True, dtype=tf.float32, name='w2v_emb_matrix')
    # [B, 1, D]
    w2v_itemid_embed = tf.nn.embedding_lookup(w2v_embedding_matrix, itemid_indices)
    w2v_itemid_embed = tf.compat.v1.layers.dense(inputs=w2v_itemid_embed, units=params['network.embedding_size'],
                                   kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                 kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01))
                                 
    deepfm_feature = deepfm(deepfm_indices, deepfm_values, mode, params, w2v_itemid_embed)

    embedding_matrix = tf.compat.v1.get_variable('embedding_matrix',
                                                 [params['data.feature_size'], params['network.embedding_size']],
                                                 dtype=tf.float32,
                                                 initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
                                                 
    #w2v_embedding_matrix = tf.Variable(params["w2v_vec"], trainable=True, dtype=tf.float32, name='w2v_emb_matrix')
   
    # [B, 1, D]
    itemid_embed = tf.nn.embedding_lookup(w2v_embedding_matrix, itemid_indices)
    #print("@@@2", itemid_embed.shape)
    # [B, SEQ_LENGTH, D]
    itemid_embed_tile = tf.tile(itemid_embed, [1, tf.shape(click_behavior_itemid_indices)[1], 1])
    
    w2v_itemid_embed1 = tf.nn.embedding_lookup(w2v_embedding_matrix, itemid_indices)
    # [B, SEQ_LENGTH, D]
    w2v_itemid_embed_tile = tf.tile(w2v_itemid_embed1, [1, tf.shape(click_behavior_itemid_indices)[1], 1])

    # [B, 1, D]
    channelid_embed = tf.nn.embedding_lookup(embedding_matrix, channelid_indices)
    # [B, SEQ_LENGTH, D]
    channelid_embed_tile = tf.tile(channelid_embed, [1, tf.shape(click_behavior_itemid_indices)[1], 1])
    # [B, SEQ_LENGTH, D']
    item_embedding = tf.concat([itemid_embed_tile, channelid_embed_tile, w2v_itemid_embed_tile], axis=2)
    # [B, SEQ_LENGTH, D]
    behavior_itemid_embed = tf.nn.embedding_lookup(w2v_embedding_matrix, click_behavior_itemid_indices)
    
    w2v_behavior_itemid_embed = tf.nn.embedding_lookup(w2v_embedding_matrix, click_behavior_itemid_indices)
    
    # [B, SEQ_LENGTH, D]
    behavior_channelid_embed = tf.nn.embedding_lookup(embedding_matrix, click_behavior_channelid_indices)
    # [B, SEQ_LENGTH, D']
    behavior_embedding = tf.concat([behavior_itemid_embed, behavior_channelid_embed, w2v_behavior_itemid_embed], axis=2)
    # [B, SEQ_LENGTH, D]
    content_embedding = tf.nn.embedding_lookup(embedding_matrix, click_behavior_clicknum_indices)
    # [B, 1]
    sequence_length = tf.gather(click_behavior_itemid_values, [0], axis=1)
    # [B, 1, D]
    user_vector = din_layer("itemid_attention", item_embedding, behavior_embedding, sequence_length, content_embedding)

    itemid_inner, itemid_cosin, itemid_diff, itemid_diff_square = cosine_similarity(user_vector, tf.concat([itemid_embed, channelid_embed, w2v_itemid_embed1], axis=-1))

    feature = tf.concat([deepfm_feature,
                         tf.squeeze(user_vector,  axis=1), 
                         tf.squeeze(itemid_embed, axis=1),
                         tf.squeeze(channelid_embed, axis=1),
                         tf.squeeze(w2v_itemid_embed1, axis=1),
                         tf.squeeze(itemid_inner, axis=1),
                         tf.squeeze(itemid_cosin, axis=1),
                         tf.squeeze(itemid_diff,  axis=1),
                         tf.squeeze(itemid_diff_square, axis=1)], axis=1)
    return feature, domain_indicator


def deepfm(feature_indices, feature_values, mode, params, w2v_itemid_embed):
    # ============= input =============
    # (feature_indices = feature_values = [batch_size, field_size], labels = [batch_size, 1])
    # ============= init feature embeddings for FM second order part and Deep part =============
    # (feature_embeddings = [batch_size, field_size, embedding_size])

    with tf.compat.v1.variable_scope('deepfm_variables', reuse=tf.compat.v1.AUTO_REUSE):
        embedding_matrix = tf.compat.v1.get_variable('embedding_matrix',
                                                     [params['data.feature_size'], params['network.embedding_size']],
                                                     dtype=tf.float32,
                                                     initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))

    #feature_indices1 = tf.gather(params=feature_indices, indices=[0], axis=1)i
    feature_indices2 = tf.gather(params=feature_indices, indices=[i for i in range(1,len(params['index.deepfm']))], axis=1)
    #feature_embeddings1 = tf.nn.embedding_lookup(params=w2v_itemid_embed, ids=feature_indices1)
    feature_embeddings2 = tf.nn.embedding_lookup(params=embedding_matrix, ids=feature_indices2)
    feature_embeddings = tf.concat([w2v_itemid_embed, feature_embeddings2], axis=1)
    
    feature_values = tf.expand_dims(feature_values, axis=-1)
    feature_embeddings = tf.multiply(feature_embeddings, feature_values, name='feature_embeddings')
    feature_embeddings = tf.concat([feature_embeddings, w2v_itemid_embed], axis=1)

    # ============= FM part =============
    # FM first order part (first_order_output = [batch_size, 1])
    first_order_matrix = tf.Variable(tf.random.normal([params['data.feature_size'], 1], 0.0, 1.0),
                                     name='first_order_matrix')
    first_order_output = tf.nn.embedding_lookup(params=first_order_matrix, ids=feature_indices)

    #first_order_output = tf.reduce_sum(tf.multiply(first_order_output, feature_values), axis=2,
    #                                   name='first_order_output')
    origin = tf.multiply(first_order_output, feature_values)
    w2v_itemid_embed_low = tf.compat.v1.layers.dense(inputs=w2v_itemid_embed, units=1,
                                     kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                     name='w2v_itemid_embed_low')
    w2v_origin = tf.concat([origin, w2v_itemid_embed_low], axis=1)
    first_order_output = tf.reduce_sum(w2v_origin, axis=2,
                                        name='first_order_output')


    # FM second order part (second_order_output = [batch_size, 1])
    square_of_sum = tf.square(tf.reduce_sum(feature_embeddings, axis=1, keepdims=None))
    sum_of_square = tf.reduce_sum(tf.square(feature_embeddings), axis=1, keepdims=None)
    second_order_output = tf.multiply(0.5, tf.subtract(square_of_sum, sum_of_square), name='second_order_output')

    # ============= Deep part =============
    net = tf.reshape(feature_embeddings, shape=[-1, (len(params['index.deepfm'])+1)  * params['network.embedding_size']])

    for idx, units in enumerate(params['network.hidden_units'], start=1):
        net = tf.compat.v1.layers.dense(inputs=net, units=units,
                                        kernel_initializer=tf.compat.v1.keras.initializers.he_normal(),
                                        kernel_regularizer=tf.compat.v1.keras.regularizers.l2(0.01),
                                        name='deepfm_dense_layer_{}'.format(idx))
        if params['network.batch_norm']:
            net = tf.compat.v1.layers.batch_normalization(inputs=net, training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                          name='deepfm_bn_layer_{}'.format(idx))
        net = tf.nn.relu(net, name='deepfm_relu_layer_{}'.format(idx))
        if params['network.dropout'] and params['network.dropout_rate'] > 0:
            net = tf.compat.v1.layers.dropout(inputs=net, rate=params['network.dropout_rate'],
                                              training=(mode == tf.estimator.ModeKeys.TRAIN),
                                              name='deepfm_dropout_layer_{}'.format(idx))

    # ============= DeepFM network output =============
    output = tf.concat([first_order_output, second_order_output, net], axis=1)

    return output




def din_layer(feature_tag, item_embed, seq_embed, seq_length, content_embed):

    # [B,SEQ_LENGTH,D']
    attention_layer = tf.concat(
            [item_embed, seq_embed, seq_embed - item_embed,
             seq_embed * item_embed, content_embed], axis=-1)

    attention_layer = tf.compat.v1.layers.dense(attention_layer, 1, None,
                                                kernel_initializer=tf.compat.v1.keras.initializers.glorot_normal(),
                                                name=feature_tag + "attention")  # B * SEQ_LENGTH * 1

    attention_layer = tf.transpose(attention_layer, [0, 2, 1])  # B * 1 * SEQ_LENGTH

    attention_layer = attention_layer / 0.3

    facts_length = tf.reshape(seq_length, [tf.shape(seq_length)[0]])  # [B]

    attention_mask = tf.sequence_mask(facts_length, tf.shape(seq_embed)[1])  # [B, SEQ_LENGTH]

    attention_mask = tf.expand_dims(attention_mask, axis=1)  # [B, 1, SEQ_LENGTH]

    paddings = tf.ones_like(attention_layer) * (-2 ** 32 + 1)

    attention_layer = tf.where(attention_mask, attention_layer, paddings)  # [B, 1, SEQ_LENGTH]

    attention_layer = tf.nn.softmax(attention_layer)

    # [B, 1, D]
    user_vector = tf.matmul(attention_layer, seq_embed)

    return user_vector

def star(net, units, domain_indicator, params, name) :
    input_size = net.get_shape().as_list()[-1]
    hidden_units = [int(input_size)] + units
    ## 共享FCN权重
    shared_kernels = [add_weight(name=name + '_shared_kernel_' + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1])) for i in range(len(hidden_units)-1)]    

    domain_kernels = [[add_weight(name=name + '_domain_kernel_' + str(index) + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1])) for i in range(len(hidden_units) -1)] for index in range(params['network.num_domains'])]

    deep_input = net
    output_list = [net] * params['network.num_domains'] 
    for i in range(len(units)):
        for j in range(params['network.num_domains']):
            # 网络的权重由共享FCN和其domain-specific FCN的权重共同决定
            output_list[j] = tf.nn.relu(tf.tensordot(
                output_list[j], shared_kernels[i] * domain_kernels[j][i], axes=(-1, 0)))
 
    net = tf.reduce_sum(tf.stack(output_list, axis=1) * tf.expand_dims(tf.one_hot(tf.reshape(domain_indicator, [-1]), params['network.num_domains']), axis = -1), axis=1)
    return net

def add_weight(name, shape):
    embedding_matrix = tf.compat.v1.get_variable(name,
                                                 shape,
                                                 dtype=tf.float32,
                                                 initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
    return embedding_matrix

def cosine_similarity(user_vector, poi_embedding):
    inner = tf.reduce_sum(user_vector * poi_embedding, axis=2, keepdims=True)  # [B,1,E]
    cosin = inner / tf.expand_dims(
        ((tf.norm(user_vector, ord=2, axis=2) + 0.0000001) * (tf.norm(poi_embedding, ord=2, axis=2) + 0.00000001)),
        -1)
    diff = tf.abs(user_vector - poi_embedding)

    return inner, cosin, diff, diff * diff
