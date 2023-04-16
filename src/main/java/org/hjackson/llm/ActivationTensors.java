
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