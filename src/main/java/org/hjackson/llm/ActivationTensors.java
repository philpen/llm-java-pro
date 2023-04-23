
package org.hjackson.llm;
import java.util.*;
public class ActivationTensors {
  public final float[] mem;
  private final int encoded;
  private final int encoded_size; // (B, T, C)
  private final int ln1_size;
  private final int ln1;
  private final int ln1_mean_size;
  private final int ln1_mean;
  private final int ln1_rstd_size;
  private final int ln1_rstd;
  private final int qkv_size;
  private final int qkv;
  private final int atty_size;
  private final int atty;
  private final int preatt_size;
  private final int preatt;
  private final int att_size;
  private final int att;
  private final int attproj_size;
  private final int attproj;
  private final int residual2_size;
  private final int residual2;
  private final int ln2_size;
  private final int ln2;
  private final int ln2_mean_size;
  private final int ln2_mean;
  private final int ln2_rstd_size;
  private final int ln2_rstd;
  private final int fch_size;
  private final int fch;
  private final int fch_gelu_size;
  private final int fch_gelu;
  private final int fcproj_size;
  private final int fcproj;
  private final int residual3_size;
  private final int residual3;
  private final int lnf_size;
  private final int lnf;
  private final int lnf_mean_size;
  private final int lnf_mean;
  private final int lnf_rstd_size;
  private final int lnf_rstd;
  private final int logits_size;
  private final int logits;
  private final int probs_size;
  private final int probs;
  private final int losses_size;
  private final int losses;
  private final int num_activations;
  private final Map<Integer, Float> tracking = new HashMap<>();
  public ActivationTensors(GPT2Config config, int B, int T) {
    int Vp = config.padded_vocab_size;
    int L = config.num_layers;
    int NH = config.num_heads;
    int C = config.channels;
    encoded_size = B * T * C; // encoded
    encoded = 0;
    ln1_size = L * B * T * C; // ln1
    ln1 = encoded + encoded_size;
    ln1_mean_size = L * B * T;  // ln1_mean
    ln1_mean = ln1 + ln1_size;
    ln1_rstd_size = L * B * T;  // ln1_rstd
    ln1_rstd = ln1_mean + ln1_mean_size;
    qkv_size = L * B * T * 3 * C; // qkv
    qkv = ln1_rstd + ln1_rstd_size;
    atty_size = L * B * T * C;  // atty
    atty = qkv + qkv_size;
    preatt_size = L * B * NH * T * T;  // preatt
    preatt = atty + atty_size;
    att_size = L * B * NH * T * T;  // att
    att = preatt + preatt_size;
    attproj_size = L * B * T * C; // attproj
    attproj = att + att_size;
    residual2_size = L * B * T * C; // residual2
    residual2 = attproj + attproj_size;
    ln2_size = L * B * T * C; // ln2
    ln2 = residual2 + residual2_size;
    ln2_mean_size = L * B * T; // ln2_mean
    ln2_mean = ln2 + ln2_size;
    ln2_rstd_size = L * B * T; // ln2_rstd
    ln2_rstd = ln2_mean + ln2_mean_size;
    fch_size = L * B * T * 4 * C; // fch
    fch = ln2_rstd + ln2_rstd_size;
    fch_gelu_size = L * B * T * 4 * C; // fch_gelu
    fch_gelu = fch + fch_size;
    fcproj_size = L * B * T * C; // fcproj
    fcproj = fch_gelu + fch_gelu_size;
    residual3_size = L * B * T * C; // residual3
    residual3 = fcproj + fcproj_size;