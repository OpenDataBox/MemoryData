package com.linkpoket.source.domain.aisearch.entity;

import org.springframework.data.domain.Persistable;

import com.linkpoket.source.global.entity.BaseEntity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.PostLoad;
import jakarta.persistence.PostPersist;
import jakarta.persistence.Table;
import jakarta.persistence.Transient;
import jakarta.persistence.UniqueConstraint;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 아이템 본문 추출 결과 엔티티입니다.
 * 식별자를 애플리케이션이 직접 채번하므로 Persistable을 구현하여, Hibernate가 신규 저장을
 * merge(UPDATE)로 오판해 INSERT가 누락되는 것을 방지합니다.
 */
@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(
	name = "ai_item_body",
	uniqueConstraints = {
		@UniqueConstraint(name = "uk_item", columnNames = {"item_type", "item_id"})
	}
)
public class AiItemBody extends BaseEntity implements Persistable<Long> {

	@Id
	@Column(name = "id", nullable = false)
	private Long id;

	@Transient
	@Getter(AccessLevel.NONE)
	private boolean newEntity = true;

	@Column(name = "item_type", length = 20, nullable = false)
	private String itemType;

	@Column(name = "item_id", nullable = false)
	private Long itemId;

	@Column(name = "body_text", columnDefinition = "MEDIUMTEXT")
	private String bodyText;

	@Enumerated(EnumType.STRING)
	@Column(name = "extraction_status", length = 20, nullable = false)
	private ExtractionStatus extractionStatus;

	@Column(name = "fail_reason", length = 255)
	private String failReason;

	@Column(name = "retry_count", nullable = false)
	private Integer retryCount;

	@Column(name = "source_hash", length = 64)
	private String sourceHash;

	@Builder
	public AiItemBody(
		final Long id,
		final String itemType,
		final Long itemId,
		final ExtractionStatus extractionStatus
	) {
		this.id = id;
		this.itemType = itemType;
		this.itemId = itemId;
		this.extractionStatus = extractionStatus;
		this.retryCount = 0;
	}

	public void markAsDone(
		final String bodyText,
		final String sourceHash
	) {
		this.bodyText = bodyText;
		this.extractionStatus = ExtractionStatus.DONE;
		this.sourceHash = sourceHash;
		this.failReason = null;
	}

	public void markAsFailed(final String failReason) {
		this.extractionStatus = ExtractionStatus.FAILED;
		this.failReason = failReason;
		this.retryCount = this.retryCount == null ? 1 : this.retryCount + 1;
	}

	@Override
	public boolean isNew() {
		return newEntity;
	}

	@PostPersist
	@PostLoad
	void markNotNew() {
		this.newEntity = false;
	}
}
