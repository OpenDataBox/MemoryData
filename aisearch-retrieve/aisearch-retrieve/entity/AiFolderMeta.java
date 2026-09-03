package com.linkpoket.source.domain.aisearch.entity;

import com.linkpoket.source.global.entity.BaseEntity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Version;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 폴더별 AI 메타데이터 엔티티입니다.
 */
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "ai_folder_meta")
public class AiFolderMeta extends BaseEntity {

	@Id
	@Column(name = "folder_id", nullable = false)
	private Long folderId;

	@Column(name = "description", columnDefinition = "TEXT", nullable = false)
	private String description;

	@Column(name = "raw_title_index", columnDefinition = "TEXT", nullable = false)
	private String rawTitleIndex;

	@Column(name = "embedding")
	private byte[] embedding;

	@Version
	@Column(name = "version", nullable = false)
	private Long version;

	@Builder
	public AiFolderMeta(
		final Long folderId,
		final String description,
		final String rawTitleIndex,
		final byte[] embedding
	) {
		this.folderId = folderId;
		this.description = description;
		this.rawTitleIndex = rawTitleIndex;
		this.embedding = embedding;
	}

	public void updateDescriptionAndEmbedding(
		final String description,
		final byte[] embedding
	) {
		this.description = description;
		this.embedding = embedding;
	}

	public void updateRawTitleIndex(final String rawTitleIndex) {
		this.rawTitleIndex = rawTitleIndex;
	}
}
