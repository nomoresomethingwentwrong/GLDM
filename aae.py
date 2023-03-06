from model import AbstractModel
from model_utils import GenericMLP, MoLeROutput, PropertyRegressionMLP, DiscriminatorMLP
from encoder import GraphEncoder, PartialGraphEncoder
from decoder import MLPDecoder
import torch

"""
1. Clone the conda env moler-env => try to pip install fcd
2. write callback function/on_validation_end during validation step that logs the
molecules from the decoded random latent vectors => add in the computation of
the fcd and log it as a metric

https://github.com/insilicomedicine/fcd_torch
https://github.com/ebartrum/lightning_gan_zoo/blob/main/run_network.py (shows that fid is monitored in the case of gans)
https://github.com/ebartrum/lightning_gan_zoo/blob/main/core/lightning_module.py
"""

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# https://github.com/P2333/Bag-of-Tricks-for-AT
# https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/wgan.py
# https://github.com/bfarzin/pytorch_aae/blob/master/main_aae.py
class AAE(AbstractModel):
    def __init__(
        self,
        params,
        dataset,
        using_lincs,
        include_predict_gene_exp_mlp=False,
        num_train_batches=1,
        batch_size=1,
    ):
        """Params is a nested dictionary with the relevant parameters."""
        super(AAE, self).__init__()
        self._init_params(params, dataset)
        self.save_hyperparameters()
        if "training_hyperparams" in params:
            self._training_hyperparams = params["training_hyperparams"]
        else:
            self._training_hyperparams = None
        self._params = params
        self._num_train_batches = num_train_batches
        self._batch_size = batch_size
        self._use_oclr_scheduler = params["use_oclr_scheduler"]
        self._decode_on_validation_end = params["decode_on_validation_end"]
        self._using_cyclical_anneal = params["using_cyclical_anneal"]
        # Graph encoders
        self._full_graph_encoder = GraphEncoder(**self._params["full_graph_encoder"])
        self._partial_graph_encoder = PartialGraphEncoder(
            **self._params["partial_graph_encoder"]
        )

        # Replace this with any other latent space mapping techniques eg diffusion
        self._latent_repr_mlp = GenericMLP(**self._params["latent_repr_mlp"])

        # MLP for regression task on graph properties
        self._include_property_regressors = "graph_properties" in self._params
        if self._include_property_regressors:
            self._graph_property_pred_loss_weight = self._params[
                "graph_property_pred_loss_weight"
            ]
            self._property_predictors = torch.nn.ModuleDict()
            for prop_name, prop_params in self._params["graph_properties"].items():
                prop_stddev = dataset.metadata.get(f"{prop_name}_stddev")
                if not (prop_params.get("normalise_loss", True)):
                    prop_stddev = None
                self._property_predictors[prop_name] = PropertyRegressionMLP(
                    **prop_params["mlp"],
                    property_stddev=prop_stddev,
                )

        # MLP decoders
        self._decoder = MLPDecoder(self._params["decoder"])

        # params for latent space
        self._latent_sample_strategy = self._params["latent_sample_strategy"]
        self._latent_repr_dim = self._params["latent_repr_size"]
        self._kl_divergence_weight = self._params["kl_divergence_weight"]
        self._kl_divergence_annealing_beta = self._params[
            "kl_divergence_annealing_beta"
        ]
        self._generator = torch.nn.ModuleList(
            [
                self._full_graph_encoder,
                self._partial_graph_encoder,
                # self._mean_log_var_mlp,
                self.latent_repr_mlp,
                self._property_predictors,
                self._decoder,
            ]
        )
        # Discriminator
        self._discriminator = DiscriminatorMLP(**self._params["discriminator"])

        # If using lincs gene expression
        self._using_lincs = using_lincs
        self._include_predict_gene_exp_mlp = include_predict_gene_exp_mlp
        if self._using_lincs:
            self._gene_exp_condition_mlp = GenericMLP(
                **self._params["gene_exp_condition_mlp"]
            )
            if self._include_predict_gene_exp_mlp:
                self._gene_exp_prediction_mlp = PropertyRegressionMLP(
                    **self._params["gene_exp_prediction_mlp"]
                )

    @property
    def latent_repr_mlp(self):
        return self._latent_repr_mlp

    @property
    def discriminator(self):
        return self._discriminator

    def _init_params(self, params, dataset):
        """
        Initialise class weights for next node prediction and placefolder for
        motif/node embeddings.
        """

        self._motif_vocabulary = dataset.metadata.get("motif_vocabulary")
        self._uses_motifs = self._motif_vocabulary is not None

        self._node_categorical_num_classes = len(dataset.node_type_index_to_string)

        if self.uses_categorical_features:
            if "categorical_features_embedding_dim" in params:
                self._node_categorical_features_embedding = None

        if self.uses_motifs:
            # Record the set of atom types, which will be a subset of all node types.
            self._atom_types = set(
                dataset._atom_type_featuriser.index_to_atom_type_map.values()
            )

        self._index_to_node_type_map = dataset.node_type_index_to_string
        self._atom_featurisers = dataset._metadata["feature_extractors"]
        self._num_node_types = dataset.num_node_types

    # def sample_from_latent_repr(self, latent_repr):
    #     mean_and_log_var = self.mean_log_var_mlp(latent_repr)
    #     # mean_and_log_var = torch.clamp(mean_and_log_var, min=-10, max=10)
    #     # perturb latent repr
    #     mu = mean_and_log_var[:, : self.latent_dim]  # Shape: [V, MD]
    #     log_var = mean_and_log_var[:, self.latent_dim :]  # Shape: [V, MD]

    #     # result_representations: shape [num_partial_graphs, latent_repr_dim]
    #     z = self.reparametrize(mu, log_var)
    #     # p, q, z = self.reparametrize(mu, log_var)

    #     return mu, log_var, z
    #     # return p, q, z

    # def reparametrize(self, mu, log_var):
    #     """Samples a different noise vector for each partial graph.
    #     TODO: look into the other sampling strategies."""
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu
    #     p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    #     q = torch.distributions.Normal(mu, std)
    #     z = q.rsample()
    #     return p, q, z

    def condition_on_gene_expression(self, latent_representation, gene_expressions):
        """
        Latent representation has size batch_size x latent_dim
        Gene expressions have size batch_size x 978
        Output dimensions is batch_size x latent_dim
        """
        return self._gene_exp_condition_mlp(
            torch.cat((latent_representation, gene_expressions), dim=-1)
        )

    def forward(self, batch):
        moler_output = self._run_step(batch)
        return (
            moler_output.first_node_type_logits,
            moler_output.node_type_logits,
            moler_output.edge_candidate_logits,
            moler_output.edge_type_logits,
            moler_output.attachment_point_selection_logits,
        )

    def _run_step(self, batch):
        # Obtain graph level representation of original molecular graph
        input_molecule_representations = self.full_graph_encoder(
            original_graph_node_categorical_features=batch.original_graph_node_categorical_features,
            node_features=batch.original_graph_x.float(),
            edge_index=batch.original_graph_edge_index,
            edge_features=batch.original_graph_edge_features,  # can be edge_type or edge_attr
            batch_index=batch.original_graph_x_batch,
        )

        # Obtain graph level representation of the partial graph
        partial_graph_representions, node_representations = self.partial_graph_encoder(
            partial_graph_node_categorical_features=batch.partial_node_categorical_features,
            node_features=batch.x,
            edge_index=batch.edge_index.long(),
            edge_features=batch.partial_graph_edge_features,
            graph_to_focus_node_map=batch.focus_node,
            candidate_attachment_points=batch.valid_attachment_point_choices,
            batch_index=batch.batch,
        )

        """
        Here, when we were using the vae, we mapped to 1024, half of it represented the mean, half represented the std.
        so at the end of the day, we still arrived at a dim of 512. However, now we have a choice, to simply map it to 512,
        which requires us to change the output of the latent_repr_mlp to map to 512 instead of 1024 
        """
        if self._using_lincs:
            latent_representation = self.condition_on_gene_expression(
                latent_representation=input_molecule_representations,
                gene_expressions=batch.gene_expressions,
            )  # currently maps to 512
        else:
            latent_representation = self.latent_repr_mlp(
                input_molecule_representations
            )  # currently maps to 512

        # Forward pass through decoder
        (
            first_node_type_logits,
            node_type_logits,
            edge_candidate_logits,
            edge_type_logits,
            attachment_point_selection_logits,
        ) = self.decoder(
            input_molecule_representations=latent_representation,
            graph_representations=partial_graph_representions,
            graphs_requiring_node_choices=batch.correct_node_type_choices_batch.unique(),
            # edge selection
            node_representations=node_representations,
            num_graphs_in_batch=len(batch.ptr) - 1,
            focus_node_idx_in_batch=batch.focus_node,
            node_to_graph_map=batch.batch,
            candidate_edge_targets=batch.valid_edge_choices[:, 1].long(),
            candidate_edge_features=batch.edge_features,
            # attachment selection
            candidate_attachment_points=batch.valid_attachment_point_choices.long(),
        )

        # NOTE: loss computation will be done in lightning module
        return MoLeROutput(
            first_node_type_logits=first_node_type_logits,
            node_type_logits=node_type_logits,
            edge_candidate_logits=edge_candidate_logits,
            edge_type_logits=edge_type_logits,
            attachment_point_selection_logits=attachment_point_selection_logits,
            mu=torch.randn_like(
                latent_representation
            ),  # placeholder; not actually used
            log_var=torch.randn_like(
                latent_representation
            ),  # placeholder; not actually used
            # p=p,
            # q=q,
            latent_representation=latent_representation,
        )

    def compute_loss(self, moler_output, batch, optimizer_idx):
        """
        If optimizer == 0, we train the generator only:
        optimizer 0 will contain every parameter other than the discriminator.
        As per normal for adversarial training, we will pass it throught the GNN encoders
        and take the latent space and pass it to the discriminator, forcing the discriminator
        to predict a label of "truth" ie 1 (BCE). During this step, the discriminator itself
        will not be updated since the parameters are not in optimizer 0.
        At the same time, we also compute the reconstruction loss from the decoders.


        If optimizer == 1, we train the discriminator only:
        optimizer 1 will contain only the discriminator parameters.
        As per normal for adversarial training, we pass the real latent vectors from the GNN
        encoders to the discriminator along with a bunch of fake latent vectors sampled
        from a normal distribution. With that, we also give the discriminator the labels for
        the real (1) and fake (0) vectors. During this step, the discriminator will be updated,
        but none of the other parts of the model will be updated.

        """
        # num_correct_node_type_choices = (
        #     batch.correct_node_type_choices_ptr.unique().shape[-1] - 1
        # )
        node_type_multihot_labels = batch.correct_node_type_choices  # .view(
        #     num_correct_node_type_choices, -1
        # )

        first_node_type_multihot_labels = (
            batch.correct_first_node_type_choices
        )  # .view(len(batch.ptr) -1, -1)

        loss = {}
        # this step only updates the generator parameters and leaves the discriminator
        # untouched during the weight update
        if optimizer_idx == 0:
            predictions_real_latents = self.discriminator(
                moler_output.latent_representation
            )
            loss["adversarial_loss"] = self.discriminator.compute_loss(
                predictions=predictions_real_latents,
                labels=torch.ones_like(
                    predictions_real_latents,
                    device=self.full_graph_encoder._dummy_param.device,
                ),
            )  # recall that the discriminator predicts 1 for real and 0 for fake

            # reconstruction loss
            loss["decoder_loss"] = self.decoder.compute_decoder_loss(
                # node selection
                node_type_logits=moler_output.node_type_logits,
                node_type_multihot_labels=node_type_multihot_labels,
                # first node selection
                first_node_type_logits=moler_output.first_node_type_logits,
                first_node_type_multihot_labels=first_node_type_multihot_labels,
                # edge selection
                num_graphs_in_batch=len(batch.ptr) - 1,
                node_to_graph_map=batch.batch,
                candidate_edge_targets=batch.valid_edge_choices[:, 1].long(),
                edge_candidate_logits=moler_output.edge_candidate_logits,  # as is
                per_graph_num_correct_edge_choices=batch.num_correct_edge_choices,
                edge_candidate_correctness_labels=batch.correct_edge_choices,
                no_edge_selected_labels=batch.stop_node_label,
                # edge type selection
                correct_edge_choices=batch.correct_edge_choices,
                valid_edge_types=batch.valid_edge_types,
                edge_type_logits=moler_output.edge_type_logits,
                edge_type_onehot_labels=batch.correct_edge_types,
                # attachement point
                attachment_point_selection_logits=moler_output.attachment_point_selection_logits,
                attachment_point_candidate_to_graph_map=batch.valid_attachment_point_choices_batch.long(),
                attachment_point_correct_choices=batch.correct_attachment_point_choice.long(),
            )
        # here we only update the discriminator parameters and nothing else
        elif optimizer_idx == 1:
            predictions_real_latents = self.discriminator(
                moler_output.latent_representation
            )

            fake_latent_vectors = torch.randn_like(
                moler_output.latent_representation,
                device=self.full_graph_encoder._dummy_param.device,
            )
            predictions_fake_latents = self.discriminator(fake_latent_vectors)
            loss["adversarial_loss"] = self.discriminator.compute_loss(
                predictions=torch.cat(
                    (
                        predictions_real_latents,
                        predictions_fake_latents,
                    ),
                    dim=0,
                ),  # concat along batch dim
                labels=torch.cat(
                    (
                        torch.ones_like(
                            predictions_real_latents,
                            device=self.full_graph_encoder._dummy_param.device,
                        ),
                        torch.zeros_like(
                            predictions_fake_latents,
                            device=self.full_graph_encoder._dummy_param.device,
                        ),
                    ),
                    dim=0,
                ),  # recall that the discriminator predicts 1 for real and 0 for fake
            )
        return loss

    def compute_property_prediction_loss(self, latent_representation, batch):
        """TODO: Since graph property regression is more of a auxillary loss than anything, this function will be
        decoupled in the future into `compute_properties` and `compute_property_prediction_loss` so that
        it can be passed into the `_run_step` function and returned in MolerOutput."""
        property_prediction_losses = {}
        for prop_name, property_predictor in self._property_predictors.items():
            predictions = property_predictor(latent_representation)
            property_prediction_losses[prop_name] = property_predictor.compute_loss(
                predictions=predictions, labels=batch[prop_name]
            )
        # sum up all the property prediction losses
        return sum([loss for loss in property_prediction_losses.values()])

    def compute_gene_expression_prediction_loss(self, latent_representation, batch):
        predictions = self._gene_exp_prediction_mlp(latent_representation)
        gene_expression_prediction_loss = self._gene_exp_prediction_mlp.compute_loss(
            predictions=predictions, labels=batch.gene_expressions
        )
        return gene_expression_prediction_loss

    def step(self, batch, optimizer_idx):
        moler_output = self._run_step(batch)

        loss_metrics = {}
        losses = self.compute_loss(
            moler_output=moler_output, batch=batch, optimizer_idx=optimizer_idx
        )

        for key in losses:
            loss_metrics[key] = losses[key]

        # NOTE: we check if we are optimizing the generator during the current step
        # if we aren't then we won't compute the property prediction regressors
        if self._include_property_regressors and optimizer_idx == 0:
            loss_metrics["property_prediction_loss"] = (
                self._graph_property_pred_loss_weight
                * self.compute_property_prediction_loss(
                    latent_representation=moler_output.latent_representation,
                    batch=batch,
                )
            )
        if self._include_predict_gene_exp_mlp:
            loss_metrics['gene_expression_prediction_loss'] = (
                self._graph_property_pred_loss_weight
                * self.compute_gene_expression_prediction_loss(
                    latent_representation=moler_output.latent_representation,
                    batch=batch,
                )
            )
        """Instead of computing the kl divergence loss, we simply allow the latent space to be decoded """
        # kld_summand = torch.square(moler_output.mu)
        # + torch.exp(moler_output.log_var)
        # - moler_output.log_var
        # - 1
        # loss_metrics['kld_loss'] = torch.mean( kld_summand)/2.0

        # annealing_factor = self.trainer.global_step % (self._num_train_batches // 4) if self._using_cyclical_anneal else self.trainer.global_step

        # loss_metrics['kld_weight'] = (
        #     (  # cyclical anealing where each cycle will span 1/4 of the training epoch
        #         1.0
        #         - self._kl_divergence_annealing_beta
        #         ** annealing_factor
        #     )
        #     * self._kl_divergence_weight
        # )

        # loss_metrics['kld_loss'] *= loss_metrics['kld_weight']

        loss_metrics["loss"] = sum(loss_metrics.values())

        logs = loss_metrics
        return loss_metrics["loss"], logs

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, logs = self.step(batch, optimizer_idx=optimizer_idx)
        for metric in logs:
            self.log(f"train_{metric}", logs[metric], batch_size=self._batch_size)
        return loss

    # def validation_step(self, batch, batch_idx, optimizer_idx):
    #     loss, logs = self.step(batch, optimizer_idx= optimizer_idx)
    #     for metric in logs:
    #         self.log(f"val_{metric}", logs[metric], batch_size=self._batch_size)
    #     return loss

    def configure_optimizers(self):
        # Separate out the discriminator params like in
        # https://github.com/airctic/icevision/issues/896
        # https://stackoverflow.com/questions/73629330/what-exactly-is-meant-by-param-groups-in-pytorch
        optimizer_gen = torch.optim.Adam(
            self._generator.parameters(), lr=self._training_hyperparams["max_lr"]
        )
        optimizer_discrim = torch.optim.Adam(
            self._discriminator.parameters(),
            lr=self._training_hyperparams["max_lr"],
        )
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self._training_hyperparams["max_lr"],
        #     betas=(0.9, 0.999),
        # )
        if self._use_oclr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer_gen,
                max_lr=self._training_hyperparams["max_lr"],
                div_factor=self._training_hyperparams["div_factor"],
                three_phase=self._training_hyperparams["three_phase"],
                epochs=self.trainer.max_epochs,
                # number of times step() is called by the scheduler per epoch
                # take the number of batches // frequency of calling the scheduler
                steps_per_epoch=self._num_train_batches // self.trainer.max_epochs,
            )

            lr_scheduler_params = {}
            lr_scheduler_params["scheduler"] = lr_scheduler

            lr_scheduler_params["interval"] = "step"
            frequency_of_lr_scheduler_step = self.trainer.max_epochs
            lr_scheduler_params[
                "frequency"
            ] = frequency_of_lr_scheduler_step  # number of batches to wait before calling lr_scheduler.step()

            # optimizer_dict = {}
            # optimizer_dict["optimizer"] = optimizer_gen
            # optimizer_dict["lr_scheduler"] = lr_scheduler_params
            return [optimizer_gen, optimizer_discrim], lr_scheduler_params
        # else:
        return [optimizer_gen, optimizer_discrim]
